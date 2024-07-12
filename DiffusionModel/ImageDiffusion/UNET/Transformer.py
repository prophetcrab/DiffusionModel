import torch
from ImageDiffusion.UNET.CrossAttention import *

class Transformer(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        #in
        self.norm_in = torch.nn.GroupNorm(
            num_groups=32,
            num_channels=dim,
            eps=1e-6,
            affine=True
        )
        self.cnn_in = torch.nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        #atten
        self.norm_atten0 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.atten1 = CrossAttention(dim, dim)
        self.norm_atten1 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.atten2 = CrossAttention(dim, 768)

        #act
        self.norm_act = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.fc0 = torch.nn.Linear(dim, dim*8)
        self.act = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(dim*4, dim)

        #out
        self.cnn_out = torch.nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, q, kv):
        #q -> [1, 320, 64, 64]
        #kv -> [1, 77, 768]
        b, _, h, w = q.shape
        res1 = q

        #-------in-------
        #维度不变
        #[1, 320, 64, 64]
        q = self.cnn_in(self.norm_in(q))

        #[1, 320, 64, 64] -> [1, 64, 64, 320] -> [1, 4096, 320]
        q = q.permute(0, 2, 3, 1).reshape(b, h*w, self.dim)

        #---attn----
        #维度不变
        #[1,4096, 320]
        q = self.atten1(q=self.norm_atten0(q), kv=self.norm_atten0(q)) + q
        q = self.atten2(q=self.norm_atten1(q), kv=kv) + q

        #----act-----------
        res2 = q
        q = self.fc0(self.norm_act(q))
        d = q.shape[2] // 2
        q = q[:, :, :d] * self.act(q[:,:,d:])

        q = self.fc1(q) + res2

        #---------out-------------
        q = q.reshape(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous()
        q = self.cnn_out(q) + res1

        return q

if __name__ == '__main__':
    print(Transformer(320)(torch.randn(1, 320, 64, 64), torch.randn(1, 77, 768)).shape)
