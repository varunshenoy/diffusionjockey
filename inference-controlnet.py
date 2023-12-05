from diffusers import StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler, ControlNetModel
from PIL import Image
import torch

controlnet = ControlNetModel.from_pretrained("./controlnet-output/checkpoint-12000/controlnet", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "riffusion/riffusion-model-v1",
        controlnet=controlnet,
        revision="main",
        torch_dtype=torch.float16,
)

pipe.to("cuda:7")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
prompt = "Electronic, Dubstep, Jazz, Cello"

dynamic = Image.open("dynamic-mask.jpg")
image = pipe(prompt=prompt, image=dynamic).images[0]
image.save("spectogram.png")

