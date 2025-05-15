import argparse
from pathlib import Path
from typing import List

from munch import munchify
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from torchvision import transforms

from util import set_seed, get_img_list, process_text
from functions.degradation import get_degradation

@torch.no_grad
def precompute(args, prompts:List[str], solver) -> List[torch.Tensor]:
    prompt_emb_set = []
    pooled_emb_set = []

    num_samples = args.num_samples if args.num_samples > 0 else len(prompts)
    for prompt in prompts[:num_samples]:
        prompt_emb, pooled_emb = solver.encode_prompt(prompt, batch_size=1)
        prompt_emb_set.append(prompt_emb)
        pooled_emb_set.append(pooled_emb)

    return prompt_emb_set, pooled_emb_set

def _get_solver(base_model, method):
    if base_model == 'flux':
        from flux_sampler import get_solver
    elif base_model == 'sd3':
        from sd3_sampler import get_solver
    else:
        raise ValueError(f"Invalid base model: {base_model}")
    return get_solver(method)

def process_prompts(solver, args):
    # load text prompts
    prompts = process_text(prompt=args.prompt, prompt_file=args.prompt_file)
    if args.base_model == "sd3":
        solver.text_enc_1.to('cuda')
        solver.text_enc_2.to('cuda')
        solver.text_enc_3.to('cuda')

        if args.efficient_memory:
            # precompute text embedding and remove encoders from GPU
            # This will allow us 1) fast inference 2) with lower memory requirement (<24GB)
            with torch.no_grad():
                prompt_emb_set, pooled_emb_set = precompute(args, prompts, solver)
                null_emb, null_pooled_emb = solver.encode_prompt([''], batch_size=1)

            del solver.text_enc_1
            del solver.text_enc_2
            del solver.text_enc_3
            torch.cuda.empty_cache()

            prompt_embs = [[x, y] for x, y in zip(prompt_emb_set, pooled_emb_set)]
            null_embs = [null_emb, null_pooled_emb]
        else:
            prompt_embs = [[None, None]] * len(prompts)
            null_embs = [None, None]
    elif args.base_model == "flux":
        solver.text_encoder.to('cuda')
        solver.text_encoder_2.to('cuda')

        if args.efficient_memory:
            # precompute text embedding and remove encoders from GPU
            # This will allow us 1) fast inference 2) with lower memory requirement (<24GB)
            with torch.no_grad():
                prompt_emb_set, pooled_emb_set = precompute(args, prompts, solver)
                null_emb, null_pooled_emb = solver.encode_prompt([''], batch_size=1)

            del solver.text_encoder
            del solver.text_encoder_2
            torch.cuda.empty_cache()

            prompt_embs = [[x, y] for x, y in zip(prompt_emb_set, pooled_emb_set)]
            null_embs = [null_emb, null_pooled_emb]
        else:
            prompt_embs = [[None, None]] * len(prompts)
            null_embs = [None, None]
    else:
        raise ValueError(f"Invalid base model: {args.base_model}")

    print("Prompts are processed.")
    return prompts, prompt_embs, null_embs

def run(args):
    # load solver
    solver = _get_solver(args.base_model, args.method)

    # process prompts
    prompts, prompt_embs, null_embs = process_prompts(solver, args)

    solver.vae.to('cuda')
    solver.transformer.to('cuda')

    # problem setup
    deg_config = munchify({
        'channels': 3,
        'image_size': args.img_size,
        'deg_scale': args.deg_scale,
        'num_blocks': args.num_blocks
    })
    operator = get_degradation(args.task, deg_config, solver.transformer.device)

    # solve problem
    tf = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor()
    ])

    pbar = tqdm(get_img_list(args.img_path), desc="Solving")
    for i, path in enumerate(pbar):
        img = tf(Image.open(path).convert('RGB'))
        img = img.unsqueeze(0).to(solver.vae.device)
        img = img * 2 - 1

        y = operator.A(img)
        y = y + 0.03 * torch.randn_like(y)
        save_image(y, args.workdir.joinpath(f'input/{str(i).zfill(4)}.png'), normalize=True)
        #exit()

        out = solver.sample(measurement=y,
                            operator=operator,
                            prompts=prompts[i] if len(prompts)>1 else prompts[0],
                            NFE=args.NFE,
                            img_shape=(args.img_size, args.img_size),
                            cfg_scale=args.cfg_scale,
                            step_size=args.step_size,
                            task=args.task,
                            prompt_emb=prompt_embs[i] if len(prompt_embs)>1 else prompt_embs[0],
                            null_emb=null_embs
                            )
        # save results
        save_image(operator.At(y).reshape(img.shape),
                   args.workdir.joinpath(f'input/{str(i).zfill(4)}.png'),
                   normalize=True)
        save_image(out,
                   args.workdir.joinpath(f'recon/{str(i).zfill(4)}.png'),
                   normalize=True)
        save_image(img,
                   args.workdir.joinpath(f'label/{str(i).zfill(4)}.png'),
                   normalize=True)

        if (i + 1) == args.num_samples:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # sampling params
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--NFE', type=int, default=28)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--img_size', type=int, default=768)

    # workdir params
    parser.add_argument('--workdir', type=Path, default='workdir')

    # data params
    parser.add_argument('--img_path', type=Path)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=-1)

    # problem params
    parser.add_argument('--task', type=str, default='sr_avgpool')
    parser.add_argument('--method', type=str, default='flowdps')
    parser.add_argument('--deg_scale', type=int, default=12)
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[11, 11])

    # solver params
    parser.add_argument('--step_size', type=float, default=15.0)
    parser.add_argument('--efficient_memory',default=False, action='store_true')
    parser.add_argument('--base_model', type=str, default='flux', choices=['flux', 'sd3'])
    args = parser.parse_args()


    # workdir creation and seed setup
    set_seed(args.seed)
    args.workdir.joinpath('input').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('recon').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('label').mkdir(parents=True, exist_ok=True)

    # run main script
    run(args)

