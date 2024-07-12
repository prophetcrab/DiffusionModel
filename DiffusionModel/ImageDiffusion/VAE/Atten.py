import torch
import torch.nn as nn

class Atten(nn.Module):

    def __init__(self):
        super(Atten, self).__init__()
        self.norm = torch.nn.GroupNorm(
            num_channels=512,
            num_groups=32,
            eps=1e-6,
            affine=True
        )
        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 512)
        self.out = torch.nn.Linear(512, 512)

    def forward(self, x):
        # x -> [1, 512, 64, 64]
        res = x

        #norm, 维度不变
        #[1, 512, 64, 64]
        x = self.norm(x)

        #[1, 512, 64, 64] -> [1, 512, 4096] -> [1, 4096, 512]
        x = x.flatten(start_dim=2).transpose(1, 2)

        #线性运算，维度不变
        #[1, 4096, 512]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        #[1, 4096, 512] * [1, 512, 4096] -> [1, 4096, 4096]
        #1/512**0.5
        k = k.transpose(1, 2)

        atten = torch.baddbmm(torch.empty(1, 4096, 4096, device=q.device),
                              q,
                              k,
                              beta=0,
                              alpha=0.044194173824159216)

        atten = torch.softmax(atten, dim=2)
        #[1,4096,4096] * [1, 4096, 512] -> [1,4096,512]
        atten = atten.bmm(v)
        atten = self.out(atten)
        atten = atten.transpose(1, 2).reshape(-1, 512, 64, 64)

        #残差连接
        atten = atten + res

        return atten

if __name__ == '__main__':
    print(Atten()(torch.randn(1, 512, 64, 64)).shape)