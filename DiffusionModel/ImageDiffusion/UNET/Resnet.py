import torch
import torch.nn as nn

class Resnet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Resnet, self).__init__()

        self.time = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(1280, dim_out),
            torch.nn.Unflatten(dim=1, unflattened_size=(dim_out, 1, 1)),
        )

        self.s0 = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_in,
                               eps=1e-5,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_in,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
        )

        self.s1 = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_out,
                               eps=1e-5,
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
            self.res = torch.nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, x, time):
        #x -> [1, 320, 64, 64]
        #time -> [1, 1280]
        res = x
        #[1, 1280] -> [1, 640, 1, 1]
        time = self.time(time)
        x = self.s0(x) + time
        x = self.s1(x)

        if self.res:
            res = self.res(res)

        x = x + res
        return x

if __name__ == '__main__':
    print(Resnet(320, 640)(torch.randn(1, 320, 32, 32), torch.randn(1, 1280)).shape)