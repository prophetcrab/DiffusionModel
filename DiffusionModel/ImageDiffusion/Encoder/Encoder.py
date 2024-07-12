
from ImageDiffusion.Encoder.TextEmbedding import *
from ImageDiffusion.Encoder.ClipEncoder import *
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            TextEmbedding(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            ClipEncoder(),
            torch.nn.LayerNorm(768),
    )

    def forward(self, x):
        return self.encoder(x)

if __name__ == '__main__':
    print(Encoder()(torch.ones(2, 77).long()).shape)