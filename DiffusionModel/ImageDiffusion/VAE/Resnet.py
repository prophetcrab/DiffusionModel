import torch
import torch.nn as nn

class ResNet(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ResNet, self).__init__()

        self.s = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_in,
                               eps=1e-6,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_in,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_out,
                               eps=1e-6,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_out,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
        )

        self.res = None
        if dim_in != dim_out:
            self.res = torch.nn.Conv2d(dim_in,
                                       dim_out,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, x):
        #x -> [1, 128, 10, 10]
        res = x
        if self.res:
            res = self.res(x)

        return res + self.s(x)

if __name__ == '__main__':
    print(ResNet(128, 256)(torch.randn(1, 128, 10, 10)).shape)