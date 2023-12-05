from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = DiffusionPipeline.from_pretrained(
        "riffusion/riffusion-model-v1",
        revision="main",
        torch_dtype=torch.float16,
)
pipe.to("cuda:7")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "Piano, Cello, Jazz"

image = pipe(prompt=prompt).images[0]

image.save("spectogram.png")

