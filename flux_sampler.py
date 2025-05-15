from typing import List, Tuple, Optional
import math
import torch

from tqdm import tqdm
from diffusers import FluxPipeline


# =======================================================================
# Factory
# =======================================================================

__SOLVER__ = {}

def register_solver(name:str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name:str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

# =======================================================================


class FluxBase():
    def __init__(self, model_key:str='black-forest-labs/FLUX.1-dev', device='cuda', dtype=torch.float16):
        self.device = device
        self.dtype = dtype

        pipe = FluxPipeline.from_pretrained(model_key, torch_dtype=self.dtype)

        self.scheduler = pipe.scheduler

        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2

        self.vae = pipe.vae
        self.transformer = pipe.transformer
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)-1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # Flux latents are turned into 2x2 patches and packed
        self.patch_size = 2

        del pipe

    def encode_prompt(self, prompt: List[str], batch_size:int=1) -> List[torch.Tensor]:
        '''
        Encode prompt using Flux's text encoders
        '''
        # Process with tokenizer
        prompt_2 = prompt  # Using same prompt for both encoders
        
        # Get embeddings from CLIP model
        prompt_embeds, pooled_embeds = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1
        )
        
        # Get embeddings from T5 model
        t5_prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt_2,
            device=self.device,
            num_images_per_prompt=1
        )
        
        # Merge embeddings in the Flux format
        prompt_embeds = torch.nn.functional.pad(
            prompt_embeds, (0, t5_prompt_embeds.shape[-1] - prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([prompt_embeds, t5_prompt_embeds], dim=-2)
        
        return prompt_embeds, pooled_embeds

    def _get_clip_prompt_embeds(self, prompt, device, num_images_per_prompt=1):
        if isinstance(prompt, str):
            prompt = [prompt]
            
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = self.text_encoder(text_input_ids, output_hidden_states=False)
        
        # Use pooled output
        pooled_prompt_embeds = prompt_embeds.pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=self.dtype, device=device)
        
        # Duplicate embeddings for each generation per prompt
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(len(prompt) * num_images_per_prompt, -1)
        
        return prompt_embeds.last_hidden_state.to(dtype=self.dtype), pooled_prompt_embeds

    def _get_t5_prompt_embeds(self, prompt, device, num_images_per_prompt=1, max_sequence_length=512):
        if isinstance(prompt, str):
            prompt = [prompt]
            
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = self.text_encoder_2(text_input_ids, output_hidden_states=False)[0]
        
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=device)
        
        _, seq_len, _ = prompt_embeds.shape
        
        # Duplicate embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(len(prompt) * num_images_per_prompt, seq_len, -1)
        
        return prompt_embeds

    def initialize_latent(self, img_size:Tuple[int], batch_size:int=1, **kwargs):
        H, W = img_size
        # Ensure dimensions are compatible with Flux's packing
        latent_h = H // (self.vae_scale_factor * self.patch_size)
        latent_w = W // (self.vae_scale_factor * self.patch_size)
        latent_c = self.transformer.config.in_channels // 4

        # Initialize latent noise
        z = torch.randn(
            (batch_size, latent_c, H // self.vae_scale_factor, W // self.vae_scale_factor), 
            device=self.device, 
            dtype=self.dtype
        )
        
        # Pack latents into the Flux format
        z = self._pack_latents(z, batch_size, latent_c, H // self.vae_scale_factor, W // self.vae_scale_factor)
        
        return z

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    def _unpack_latents(self, latents, height, width):
        batch_size, num_patches, channels = latents.shape
        
        # Calculate height and width for the unpacked latents
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, height, width)
        
        return latents

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(image).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        z = z.to(dtype=self.dtype)
            
        # Pack the latents
        batch_size, latent_c, latent_h, latent_w = z.shape
        z = self._pack_latents(z, batch_size, latent_c, latent_h, latent_w)
            
        return z

    def decode(self, z: torch.Tensor, img_size: Tuple[int]) -> torch.Tensor:
        H, W = img_size
        
        # Unpack the latents
        z = self._unpack_latents(z, H, W)
        
        # Decode with VAE
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self, z, t, prompt_emb, pooled_emb):
        # Prepare latent image IDs
        batch_size, num_patches, _ = z.shape
        latent_height = latent_width = int(math.sqrt(num_patches))
        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, latent_height, latent_width, 
            self.device, self.dtype
        )
        
        # Text IDs (zeros for Flux)
        text_ids = torch.zeros(prompt_emb.shape[1], 3).to(
            device=self.device, dtype=self.dtype
        )
        
        # Convert timestep to the format expected by Flux
        timestep = t.expand(z.shape[0]).to(z.dtype) / 1000
        
        # Handle guidance if the model supports it
        guidance = None
        if hasattr(self.transformer.config, "guidance_embeds") and self.transformer.config.guidance_embeds:
            guidance = torch.full([1], 1.0, device=self.device, dtype=self.dtype)
            guidance = guidance.expand(z.shape[0])
        
        # Get prediction from transformer
        v = self.transformer(
            hidden_states=z,
            timestep=timestep,
            pooled_projections=pooled_emb,
            guidance=guidance,
            encoder_hidden_states=prompt_emb,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False
        )[0]
        
        return v

    def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_ids = latent_image_ids.reshape(height * width, 3)
        return latent_image_ids.to(device=device, dtype=dtype)


class FluxEuler(FluxBase):
    def __init__(self, model_key:str='black-forest-labs/FLUX.1-dev', device='cuda'):
        super().__init__(model_key=model_key, device=device)

    def inversion(self, src_img, prompts: List[str], NFE:int, cfg_scale: float=1.0, batch_size: int=1,
                  prompt_emb:Optional[List[torch.Tensor]]=None,
                  null_emb:Optional[List[torch.Tensor]]=None):
        '''
        Perform inversion on the source image to find a suitable latent
        '''
        img_size = src_img.shape[-2:]  # (H, W)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb = prompt_emb.to(self.transformer.device)
            pooled_emb = pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""])
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb = null_prompt_emb.to(self.transformer.device)
            null_pooled_emb = null_pooled_emb.to(self.transformer.device)

        # initialize latent from the source image
        src_img = src_img.to(device=self.vae.device, dtype=self.dtype)
        with torch.no_grad():
            z = self.encode(src_img).to(self.transformer.device)

        # timesteps 
        self.scheduler.set_timesteps(NFE, device=self.transformer.device)
        timesteps = self.scheduler.timesteps
        timesteps = torch.cat([timesteps, torch.zeros(1, device=self.transformer.device)])
        timesteps = reversed(timesteps)
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps[:-1], total=NFE, desc='Flux Euler Inversion')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.transformer.device)
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1]

            z = z + (sigma_next - sigma) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))

        return z

    def sample(self, prompts: List[str], NFE:int, img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None):
        '''
        Standard sampling from the Flux model
        '''
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb = prompt_emb.to(self.transformer.device)
            pooled_emb = pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb = null_prompt_emb.to(self.transformer.device)
            null_pooled_emb = null_pooled_emb.to(self.transformer.device)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps
        # Calculate mu for dynamic shifting
        batch_size, num_patches, _ = z.shape
        image_seq_len = num_patches
        
        # Calculate appropriate mu value based on sequence length
        base_seq_len = self.scheduler.config.get("base_image_seq_len", 256)
        max_seq_len = self.scheduler.config.get("max_image_seq_len", 4096)
        base_shift = self.scheduler.config.get("base_shift", 0.5)
        max_shift = self.scheduler.config.get("max_shift", 1.15)
        
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        
        self.scheduler.set_timesteps(NFE, device=self.device, mu=mu)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='Flux Euler')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
            if cfg_scale != 1.0:
                pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
            else:
                pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            z = z + (sigma_next - sigma) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))

        # decode
        with torch.no_grad():
            img = self.decode(z, (imgH, imgW))
        return img


@register_solver("flowdps")
class FluxFlowDPS(FluxEuler):
    def data_consistency(self, z0t, operator, measurement, task, stepsize:int=30.0):
        '''
        Apply data consistency step for the FlowDPS solver
        '''
        z0t = z0t.requires_grad_(True)
        num_iters = 3
        for _ in range(num_iters):
            # Decode latent to image space
            x0t = self.decode(z0t, operator.img_size).float()
            
            # Apply data consistency in image space
            if "sr" in task:
                loss = torch.linalg.norm((operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))).view(1, -1))
            else:
                loss = torch.linalg.norm((operator.At(measurement) - operator.At(operator.A(x0t))).view(1, -1))
                
            # Calculate gradients and update latents
            grad = torch.autograd.grad(loss, z0t)[0].half()
            z0t = z0t - stepsize * grad

        return z0t.detach()

    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_size: float=30.0,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None):
        '''
        FlowDPS solver implementation for Flux model
        '''
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb = prompt_emb.to(self.transformer.device)
            pooled_emb = pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb = null_prompt_emb.to(self.transformer.device)
            null_pooled_emb = null_pooled_emb.to(self.transformer.device)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # Set operator's image size for data consistency
        operator.img_size = (imgH, imgW)

        # timesteps
        self.scheduler.config.shift = 4.0  # Setting shift parameter for better results
        
        # Calculate mu for dynamic shifting (similar to calculate_shift function in Flux pipeline)
        batch_size, num_patches, _ = z.shape
        image_seq_len = num_patches
        
        # Calculate appropriate mu value based on sequence length
        base_seq_len = self.scheduler.config.get("base_image_seq_len", 256)
        max_seq_len = self.scheduler.config.get("max_image_seq_len", 4096)
        base_shift = self.scheduler.config.get("base_shift", 0.5)
        max_shift = self.scheduler.config.get("max_shift", 1.15)
        
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        
        # Set timesteps with the calculated mu
        self.scheduler.set_timesteps(NFE, device=self.device, mu=mu)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='Flux-FlowDPS')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)

            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)                    # Line 3: v_theta(z, c)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb) # Line 3: v_theta(z, null)
                else:
                    pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < NFE else 0.0
            vt_z = pred_null_v + cfg_scale * (pred_v - pred_null_v) # Line 3: v_t(z)

            # denoising step
            z0t = z - sigma * vt_z        # Line 4: z_0|t
            z1t = z + (1 - sigma) * vt_z  # Line 5: z_1|t

            if i < NFE:
                # Apply data consistency
                z0y = self.data_consistency(z0t, operator, measurement, task=task, stepsize=step_size)  # Line 7: z_0|t(y)
                z0y = (1 - sigma) * z0t + sigma * z0y # Line 8: z_0(t) tilde

            # renoising
            noise = math.sqrt(sigma_next) * z1t + math.sqrt(1 - sigma_next) * torch.randn_like(z1t) # Line 11: z_t tilde
            z = (1 - sigma_next) * z0y + sigma_next * noise  # Line 13: z

        # decode
        with torch.no_grad():
            img = self.decode(z, (imgH, imgW))
        return img
