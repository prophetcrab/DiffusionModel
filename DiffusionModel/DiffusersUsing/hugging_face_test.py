import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

model_id = "sd-dreambooth-library/mr-potato-head"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir='D:/Model').to(device=device)


prompt = "a cat on the wall"
image = pipe(prompt, num_inferences_steps=100, guidence_scale=7.5).images[0]

image = image.save("D:/PythonProject/DiffusionModel/Result/005.png")
print(image)
