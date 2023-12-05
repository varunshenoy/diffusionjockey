from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda:7")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("./rank512-output/checkpoint-500/pytorch_lora_weights.safetensors")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "Rock, Pop, Electronic"

image = load_image("spectogram.png")
mask_image = load_image("mask2.png")

image = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
).images[0]

image.save("spectogram-inpaint.png")

