import torch
import torch.nn as nn

class NoiseInjection(nn.Module):
    def __init__(self, weight):
        super().__init__()

        self.weight = weight

    def forward(self, image):
        batch, channel, height, width = image.shape
        noise = image.new_empty(batch, channel, height, width).normal_()
        # return image + self.weight * noise
        return image

class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        block1 = nn.Sequential(
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), padding=1),
            # nn.InstanceNorm2d(in_channel),
            # nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=0),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        block2 = nn.Sequential(
            # nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1, stride=2),
            nn.MaxPool2d(2),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        skip = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1)),
        )

        self.block1 = block1
        self.block2 = block2
        self.skip = skip
        # self.noise = NoiseInjection()


    def forward(self, x):
        x = self.block1(x) + self.skip(x)
        res = self.block2(x)
        # res = self.noise(res)
        return res


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        block1 = nn.Sequential(
            # nn.ConvTranspose2d(in_channel, in_channel, kernel_size=(3, 3), padding=(1, 1)),
            # nn.InstanceNorm2d(in_channel),
            # nn.ReLU(),
            nn.ConvTranspose2d(in_channel+out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1)),
            # nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        block2 = nn.Sequential(
            # nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=2, output_padding=1),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(),
        )
        skip = nn.Sequential(
            nn.ConvTranspose2d(in_channel+out_channel, out_channel, kernel_size=(1, 1)),
            # nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1)),
        )

        self.block1 = block1
        self.block2 = block2
        self.skip = skip
        # self.noise = NoiseInjection()


    def forward(self, r, l):

        rp = self.block2(r)

        rl = torch.cat([rp, l], dim=1)

        res = self.block1(rl) + self.skip(rl)

        return res


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        block = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        self.block = block
        self.in_channel = in_channel
        self.out_channel = out_channel
#         self.noise = NoiseInjection()

    def forward(self, x):
        if self.in_channel == self.out_channel:
            return self.block(x) + x
        res = self.block(x)
#         res = self.noise(res)
        return res


class Unet(nn.Module):
    def __init__(self, device):
        super().__init__()

        # 3 w h
        preprocess = nn.Sequential(
            # nn.ReflectionPad2d(4),
            # nn.Conv2d(3, 32, kernel_size=(9, 9)),
            # nn.Conv2d(3, 32, kernel_size=(9, 9), padding=4),
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, kernel_size=(3, 3)),
            # nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            ResBlock(32, 32),
            #             NoiseInjection(0.2),
        )
        # 32 w h
        conv1 = nn.Sequential(
            DownSample(32, 64),
            # ResBlock(64, 64),
            NoiseInjection(0.05),

        )
        # 64 w/2 h/2
        conv2 = nn.Sequential(
            DownSample(64, 128),
            # ResBlock(128, 128),
            NoiseInjection(0.05),
        )
        # 128 w/4 h/4
        conv3 = nn.Sequential(
            DownSample(128, 256),
            # ResBlock(256, 256),
            NoiseInjection(0.1),
        )
        # 256 w/8 h/8

        res = nn.Sequential(
            ResBlock(256, 256),
            NoiseInjection(0.1),
            ResBlock(256, 256),
            NoiseInjection(0.1),
        )
        res2 = nn.Sequential(
            ResBlock(128, 128),
            NoiseInjection(0.05),
            ResBlock(128, 128),
            NoiseInjection(0.05),
        )
        # 512 w/16 h/16

        upsample3 = UpSample(256, 128)

        upsample2 = UpSample(128, 64)

        upsample1 = UpSample(64, 32)


        # upsample3 = nn.Sequential(
        #     UpSample(256, 128),
        #     # ResBlock(128, 128),
        #     # NoiseInjection(0.5),
        # )
        #
        # upsample2 = nn.Sequential(
        #     UpSample(128, 64),
        #     # ResBlock(64, 64),
        #     # NoiseInjection(0.4),
        # )
        #
        # upsample1 = nn.Sequential(
        #     UpSample(64, 32),
        #     # ResBlock(32, 32),
        #     # NoiseInjection(0.3),
        # )



        postprocess = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=(3, 3), padding=1),
            # nn.Tanh(),
            nn.Sigmoid(),

        )

        # 3 w h

        self.device = device

        self.preprocess = preprocess.to(device)

        self.conv1 = conv1.to(device)
        self.conv2 = conv2.to(device)
        self.conv3 = conv3.to(device)

        self.res = res.to(device)
        self.res2 = res2.to(device)

        self.upsample1 = upsample1.to(device)
        self.upsample2 = upsample2.to(device)
        self.upsample3 = upsample3.to(device)

        self.postprocess = postprocess.to(device)

    def forward(self, x):
        x = x.to(self.device)

        # 3 512 512
        l1 = self.preprocess(x)
        # 32 512 512
        l2 = self.conv1(l1)
        # 64 256 256
        l3 = self.conv2(l2)
        # 128 128 128
        l4 = self.conv3(l3)
        # 256 64 64


        # 128 128 128
        r4 = self.res(l4)

        r3 = self.upsample3(r4, self.res2(l3))

        r2 = self.upsample2(r3, l2)

        r1 = self.upsample1(r2, l1)

        output = self.postprocess(r1)
        # output = torch.nn.functional.interpolate(output, size=(224,224), mode='bilinear')

        return output