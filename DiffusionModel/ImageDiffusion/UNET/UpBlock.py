import torch
import torch.nn as nn

from ImageDiffusion.UNET.Resnet import *
from ImageDiffusion.UNET.Transformer import *

class UpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_prev, add_up):
        super().__init__()

        self.res0 = Resnet(dim_out + dim_prev, dim_out)
        self.res1 = Resnet(dim_out + dim_out, dim_out)
        self.res2 = Resnet(dim_in + dim_out, dim_out)

        self.tf0 = Transformer(dim_out)
        self.tf1 = Transformer(dim_out)
        self.tf2 = Transformer(dim_out)

        self.out = None
        if add_up:
            self.out = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                torch.nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)
            )

    def forward(self, out_vae, out_encoder, time, out_down):
        out_vae = self.res0(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf0(out_vae, out_encoder)

        out_vae = self.res1(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf1(out_vae, out_encoder)

        out_vae = self.res2(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf2(out_vae, out_encoder)

        if self.out:
            out_vae = self.out(out_vae)

        return out_vae

if __name__ == '__main__':
    print(
        UpBlock(320, 640, 1280, True)(
            torch.randn(1, 1280, 32, 32),
            torch.randn(1, 77, 768),
            torch.randn(1, 1280),
            [
                torch.randn(1, 320, 32, 32),
                torch.randn(1, 640, 32, 32),
                torch.randn(1, 640, 32, 32)
            ]
        ).shape
    )