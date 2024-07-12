import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Atten(nn.Module):

    def __init__(self):
        super(Atten, self).__init__()
        self.q = torch.nn.Linear(768, 768)
        self.k = torch.nn.Linear(768, 768)
        self.v = torch.nn.Linear(768, 768)
        self.out = torch.nn.Linear(768, 768)

    def forward(self, x):
        #x -> [b, 77, 768]
        b = x.shape[0]

        #维度不变
        #[b, 77, 768]
        q = self.q(x) * 0.125
        k = self.k(x)
        v = self.v(x)

        #拆分注意力头
        #[b, 77, 768] -> [b, 77, 12, 64] -> [b, 12, 77, 64] -> [b*12, 77, 64]
        q = q.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)
        k = k.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)
        v = v.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)

        #计算qk乘积
        attn = torch.bmm(q, k.transpose(1, 2))

        #覆盖mask
        def get_mask(b):
            mask = torch.empty(b, 77, 77)

            #上三角部分置为负无穷，下三角部分置为0

            mask.fill_(-float('inf'))
            mask.triu_(1)

            return mask.unsqueeze(0)

        #[b, 12, 77, 77] + [b, 1, 77, 77] -> [b, 12, 77, 77]
        attn = attn + get_mask(attn.shape[0]).to(attn.device)

        #[b, 12, 77, 77] -> [b*12, 77, 77]
        attn = attn.reshape(b*12, 77, 77)

        #计算softmax,被mask的部分置为0
        attn = attn.softmax(dim=-1)

        #计算和v的乘积
        #[b*12, 77, 77] * [b*12, 77, 64] -> [b*12, 77, 64]
        attn = torch.bmm(attn, v)

        #[b*12, 77, 64] -> [b, 12, 77, 64] -> [b, 77, 12, 64] -> [b, 77, 768]
        attn = attn.reshape(b, 12, 77, 64).transpose(1, 2).reshape(b, 77, 768)

        return self.out(attn)

if __name__ == '__main__':
    print(Atten()(torch.rand(2, 77, 768)).shape)