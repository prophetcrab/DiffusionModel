import torch
import torch.nn as nn

from ImageDiffusion.VAE.Resnet import *
from ImageDiffusion.VAE.Atten import *

class Pad(nn.Module):
    def forward(self, x):
        return nn.functional.pad(x, (0, 1, 0, 1),
                                 mode='constant', value=0)
class VAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            #in
            torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),

            #down
            torch.nn.Sequential(
                ResNet(128, 128),
                ResNet(128, 128),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                ResNet(128, 256),
                ResNet(256, 256),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                ResNet(256, 512),
                ResNet(512, 512),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                ResNet(512, 512),
                ResNet(512, 512),
            ),

            #mid
            torch.nn.Sequential(
                ResNet(512, 512),
                Atten(),
                ResNet(512, 512),
            ),

            #out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(512, 8, 3, padding=1),
            ),

            #正太分布层
            torch.nn.Conv2d(8, 8, 1),
        )

        self.decoder = nn.Sequential(
            #正态分布层
            torch.nn.Conv2d(4,4,1),

            #in
            torch.nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),

            #middle
            torch.nn.Sequential(ResNet(512, 512), Atten(), ResNet(512, 512)),

            #up
            torch.nn.Sequential(
                ResNet(512, 512),
                ResNet(512, 512),
                ResNet(512, 512),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                ResNet(512, 512),
                ResNet(512, 512),
                ResNet(512, 512),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                ResNet(512, 256),
                ResNet(256, 256),
                ResNet(256, 256),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                ResNet(256, 128),
                ResNet(128, 128),
                ResNet(128, 128),
            ),

            #out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(128, 3, 3, padding=1),
            ),
        )

    def sample(self, h):
        #h -> [1, 8, 64, 64]
        #[1, 4, 64, 64]
        mean = h[:, :4]
        logvar = h[:, 4:]
        std = logvar.exp()**0.5

        #[1, 4, 64, 64]
        h = torch.randn(mean.shape, device=mean.device)
        h = mean + std * h

        return h

    def forward(self, x):


        print(x.shape)
        h = self.encoder(x)
        h = self.sample(h)
        h = self.decoder(h)

        return h

if __name__ == '__main__':
    VAE()(torch.randn(1,3,512,512))