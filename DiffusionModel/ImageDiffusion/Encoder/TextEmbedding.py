import torch
import torch.nn as nn

class TextEmbedding(nn.Module):

    def __init__(self):
        super(TextEmbedding, self).__init__()
        self.embed = torch.nn.Embedding(49408, 768)
        self.pos_embed = torch.nn.Embedding(77, 768)
        self.register_buffer('pos_ids', torch.arange(77).unsqueeze(dim=0))

    def forward(self, input_ids):
        #input_ids -> [b, 77]

        #[b,77] -> [b, 77, 768]
        embed = self.embed(input_ids)

        #[1, 77] -> [1, 77, 768]
        pos_embed = self.pos_embed(self.pos_ids)

        #[b, 77, 768]
        return embed +  pos_embed

if __name__ == '__main__':
    print(TextEmbedding()(torch.ones(2, 77).long()).shape)
