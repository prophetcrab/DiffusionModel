import torch
from ImageDiffusion.UNET.Transformer import *
from ImageDiffusion.UNET.Resnet import *

class DownBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.tf0 = Transformer(dim_out)
        self.res0 = Resnet(dim_in, dim_out)

        self.tf1 = Transformer(dim_out)
        self.res1 = Resnet(dim_out, dim_out)

        self.out = torch.nn.Conv2d(
            dim_out,
            dim_out,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, out_vae, out_encoder, time):
        outs = []

        out_vae = self.res0(out_vae, time)
        out_vae = self.tf0(out_vae, out_encoder)
        outs.append(out_vae)

        out_vae = self.res1(out_vae, time)
        out_vae = self.tf1(out_vae, out_encoder)
        outs.append(out_vae)

        out_vae = self.out(out_vae)
        outs.append(out_vae)

        return out_vae, outs

if __name__ == '__main__':
    print(
        DownBlock(320,64)(torch.randn(1, 320, 32, 32), torch.randn(1, 77, 768), torch.randn(1, 1280))[0].shape
    )