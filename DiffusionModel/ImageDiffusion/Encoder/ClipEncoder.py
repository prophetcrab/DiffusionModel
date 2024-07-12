from ImageDiffusion.Encoder.Atten import *
from ImageDiffusion.Encoder.Encoder import *

class ClipEncoder(nn.Module):

    def __init__(self):
        super(ClipEncoder, self).__init__()

        self.s1 = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            Atten(),
        )

        self.s2 = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, 3072),
        )

        self.s3 = torch.nn.Linear(3072, 768)

    def forward(self, x):
        #x -> [2, 77, 768]

        #维度不变 [2, 77, 768]
        x = x + self.s1(x)

        #[2, 77, 768]
        res = x

        #[2, 77, 768] -> [2, 77, 3072]
        x = self.s2(x)

        #维度不变
        #[2, 77, 3072]
        x = x * (x * 1.702).sigmoid()

        #[2, 77, 3072] -> [2, 77, 768]
        return res + self.s3(x)

if __name__ == '__main__':
    print(ClipEncoder()(torch.randn(2, 77, 768)).shape)