import torch
from ImageDiffusion.UNET.UpBlock import *
from ImageDiffusion.UNET.DownBlock import *
from ImageDiffusion.UNET.Resnet import *
from ImageDiffusion.UNET.Transformer import *

class UNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        #in
        self.in_vae = torch.nn.Conv2d(4, 320, kernel_size=3, padding=1)

        self.in_time = torch.nn.Sequential(
            torch.nn.Linear(320, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280),
        )

        #down
        self.down_block0 = DownBlock(320, 320)
        self.down_block1 = DownBlock(320, 640)
        self.down_block2 = DownBlock(640, 1280)

        self.down_res0 = Resnet(1280, 1280)
        self.down_res1 = Resnet(1280, 1280)

        #mid
        self.mid_res0 = Resnet(1280, 1280)
        self.mid_tf = Transformer(1280)
        self.mid_res1 = Resnet(1280, 1280)

        #up
        self.up_res0 = Resnet(2560, 1280)
        self.up_res1 = Resnet(2560, 1280)
        self.up_res2 = Resnet(2560, 1280)

        self.up_in = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(1280, 1280, kernel_size=3, padding=1),
        )

        self.up_block0 = UpBlock(640, 1280, 1280, True)
        self.up_block1 = UpBlock(320, 640, 1280, True)
        self.up_block2 = UpBlock(320, 320, 640, False)

        #out
        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(num_channels=320, num_groups=32, eps=1e-5),
            torch.nn.SiLU(),
            torch.nn.Conv2d(320, 4, kernel_size=3, padding=1),
        )

    def forward(self, out_vae, out_encoder, time):
        out_vae = self.in_vae(out_vae)

        def get_time_embed(t):
            e = torch.arange(160) * -9.210340371978184 / 160
            e = e.exp().to(t.device) * t
            e = torch.cat([e.cos(), e.sin()]).unsqueeze(dim=0)

            return e

        time = get_time_embed(time)
        time = self.in_time(time)

        out_down = [out_vae]
        out_vae, out = self.down_block0(
            out_vae=out_vae,
            out_encoder=out_encoder,
            time=time
        )
        out_down.extend(out)

        out_vae, out = self.down_block1(
            out_vae=out_vae,
            out_encoder=out_encoder,
            time=time
        )
        out_down.extend(out)

        out_vae, out = self.down_block2(
            out_vae=out_vae,
            out_encoder=out_encoder,
            time=time
        )
        out_down.extend(out)

        out_vae = self.down_res0(out_vae, time)
        out_down.append(out_vae)

        out_vae = self.down_res1(out_vae, time)
        out_down.append(out_vae)

        #---------mid----------
        out_vae = self.mid_res0(out_vae, time)
        out_vae = self.mid_tf(out_vae, out_encoder)
        out_vae = self.mid_res1(out_vae, time)

        #---------up------------
        out_vae = self.up_res0(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.up_res1(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.up_res2(torch.cat([out_vae, out_down.pop()], dim=1), time)

        out_vae = self.up_in(out_vae)

        out_vae = self.up_block0(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)

        out_vae = self.up_block1(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)

        out_vae = self.up_block2(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)

        out_vae = self.out(out_vae)

        return out_vae

if __name__ == '__main__':
    print(UNet()(
        torch.randn(2, 4, 64, 64), torch.randn(2, 77, 768), torch.LongTensor([26])
    ).shape)


