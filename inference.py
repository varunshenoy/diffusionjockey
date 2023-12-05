from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda:7")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("./rank512-output-2/checkpoint-7000/pytorch_lora_weights.safetensors")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "Piano, Cello, Jazz"

image = pipe(prompt=prompt).images[0]

image.save("spectogram.png")

