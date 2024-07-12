from diffusers import DDPMPipeline
from PIL import Image

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

butterfly = DDPMPipeline.from_pretrained("johnowhitaker/ddpm-butterflies-32px", cache_dir='D:/Model').to(device)

images = butterfly(batch_size=8).images

count = 0
for image in images:
    image.save("D:/PythonProject/DiffusionModel/Result/00{}.png".format(count))
    count = count + 1