import torch

class CrossAttention(torch.nn.Module):
    def __init__(self, dim_q, dim_kv):
        #dim_q - > 320
        #dim_kv -> 768
        super().__init__()
        self.dim_q = dim_q
        self.q = torch.nn.Linear(dim_q, dim_q, bias=False)
        self.k = torch.nn.Linear(dim_kv, dim_q, bias=False)
        self.v = torch.nn.Linear(dim_kv, dim_q, bias=False)

        self.out = torch.nn.Linear(dim_q, dim_q)

    def forward(self, q, kv):
        #x -> [1, 4096, 320]
        #kv -> [1, 77, 768]
        q = self.q(q)
        k = self.k(kv)
        v = self.v(kv)

        def reshape(x):
            b, lens, dim = x.shape
            x = x.reshape(b, lens, 8, dim//8)
            x = x.transpose(1, 2)
            x = x.reshape(b*8, lens, dim//8)
            return x

        #[1, 4096, 320] -> [8, 4096, 40]
        q = reshape(q)
        #[1, 77, 320] -> [8, 77, 40]
        k = reshape(k)
        #[1, 77, 320] -> [8, 77, 40]
        v = reshape(v)

        atten = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device),
            q,
            k.transpose(1, 2),
            beta=0,
            alpha=(self.dim_q//8)**-0.5
        )
        atten = atten.softmax(dim=-1)
        atten = atten.bmm(v)

        def reshape(x):
            b, lens, dim = x.shape
            x = x.reshape(b//8, 8, lens, dim)
            x = x.transpose(1, 2)
            x = x.reshape(b//8, lens, dim*8)

            return x

        atten = reshape(atten)
        atten = self.out(atten)

        return atten

if __name__ == '__main__':
    print(CrossAttention(320, 768)(torch.randn(1, 4096, 320), torch.randn(1, 77, 768)).shape)