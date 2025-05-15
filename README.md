# FlowDPS with FLUX

![img](assets/main.jpg)

Please refer to the original paper [FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](https://arxiv.org/abs/2503.08136) for more details. This repository is built upon the original code repository [FlowDPS](https://github.com/FlowDPS-Inverse/FlowDPS).

## Abstract


❗️Flow matching is a recent state-of-the-art framework for generative modeling based on ordinary differential equations (ODEs). While closely related to diffusion models, __it provides a more general perspective__ on generative modeling. 

❓ Although inverse problem solving has been extensively explored using diffusion models, it has not been rigorously examined within the broader context of flow models. Therefore, __we extend the diffusion inverse solvers (DIS)— which perform posterior sampling by combining a denoising diffusion prior with an likelihood gradient—into the flow framework.__

👍 Our proposed solver, Flow-Driven Posterior Sampling (FlowDPS), can also be seamlessly integrated into a latent flow model with a transformer architecture. Across four linear inverse problems, we confirm that FlowDPS outperforms state-of-the-art alternatives, all without requiring additional training.


## Quick Start

### Environment Setup

First, clone this repository and install requirements.

```
git clone https://github.com/FlowDPS-Inverse/FlowDPS.git
cd FlowDPS
conda create -n flowdps python==3.10
conda activate flowdps
pip install -r requirements.txt
```

> Provided requirements.txt installs torch with CUDA 11.8. If you are using other versions, please change it.

For the motion blur, we need to clone below repository.
```
git clone https://github.com/LeviBorodenko/motionblur.git
```

You will also need to setup a [Hugging Face](https://huggingface.co) account to be able to use the [diffusers](https://huggingface.co/docs/diffusers/en/index) library and download pretrained Stable Diffusion 3 or FLUX models. Plesae refer to their website for more details on setting up accounts and access tokens.

### Examples

You can quickly check the results using following examples.

**Example 1. Super-resolution x 12 (avg-pool) / Dog**
```
python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory;
```

**Example 2. Super-resolution x 12 (bicubic) / Animal**
```
python solve.py \
    --img_size 768 \
    --img_path samples/div2k_example.png \
    --prompt "a high quality photo of animal, bush, close-up, fox, grass, green, greenery, hide, panda, red, red panda, stare" \
    --task sr_bicubic \
    --deg_scale 12 \
    --efficient_memory;
```
> The prompt (after "a high quality photo of") is extracted by DAPE from measurement.

**Example 3. Motion Deblur / Human**
```
python solve.py \
    --img_size 768 \
    --img_path samples/ffhq_example.png \
    --prompt "a photo of a closed face" \
    --task deblur_motion \
    --deg_scale 61 \
    --efficient_memory;
```


For each task, expected results are
![expect](assets/expected.jpg)


## How to set command line arguments

You can freely change the task and solver using arguments:
- `task` : sr_avgpool / sr_bicubic / deblur_gauss / deblur_motion
- `method` : psld / flowchef / flowdps
- `base_model`: sd3 / flux. Default is flux.

If you want to change the amount of degradation, change `deg_scale`. For SR tasks, it refers to the downscale factor, and for deblurring tasks, it refers to the kernel size. 

## Efficient inference

If you use `--efficient_memory`, text encoder will pre-compute text embeddings and is removed from the GPU.

This allows us to solve inverse problem with a single GPU with VRAM 24GB.
