import torch
import torch.nn as nn

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = 1

    def forward(self, image):
        batch, channel, height, width = image.shape
        noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise

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
        self.noise = NoiseInjection()




    def forward(self, x):
        x = self.block1(x) + self.skip(x)
        res = self.block2(x)
        res = self.noise(res)
        return res

class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        block1 = nn.Sequential(
            # nn.ConvTranspose2d(in_channel, in_channel, kernel_size=(3, 3), padding=(1, 1)),
            # nn.InstanceNorm2d(in_channel),
            # nn.ReLU(),
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1)),
            # nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        block2 = nn.Sequential(
            # nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=2, output_padding=1),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        skip = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(1, 1)),
            # nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1)),
        )

        self.block1 = block1
        self.block2 = block2
        self.skip = skip

    def forward(self, x):
        x = self.block1(x) + self.skip(x)
        res = self.block2(x)
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
        # self.noise = NoiseInjection()

    def forward(self, x):
        if self.in_channel == self.out_channel:
            return self.block(x) + x
        return self.block(x)


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
        )
        # 64 w/2 h/2
        conv2 = nn.Sequential(
            DownSample(64, 128),
            # ResBlock(128, 128),
        )
        # 128 w/4 h/4
        conv3 = nn.Sequential(
            DownSample(128, 256),
            # ResBlock(256, 256),
        )
        # 256 w/8 h/8
        # conv4 = nn.Sequential(
        #     DownSample(256, 512),
        #     # ResBlock(512, 512),
        # )
        # 512 w/16 h/16

        # res = nn.Sequential(*[ResBlock(512, 512) for _ in range(2)])
        res = nn.Sequential(*[ResBlock(256, 256) for _ in range(2)])
        # 512 w/16 h/16

        # upsample4 = nn.Sequential(
        #     UpSample(512, 256),
        #     # ResBlock(256, 256),
        # )

        upsample3 = nn.Sequential(
            UpSample(256, 128),
            # ResBlock(128, 128),
        )

        upsample2 = nn.Sequential(
            UpSample(128, 64),
            # ResBlock(64, 64),
        )

        upsample1 = nn.Sequential(
            UpSample(64, 32),
            # ResBlock(32, 32),
        )

        # deconv4 = nn.Sequential(
        #     ResBlock(512, 256),
        #     ResBlock(256, 256),
        # )

        deconv3 = nn.Sequential(
            ResBlock(256, 128),
            ResBlock(128, 128),
        )

        deconv2 = nn.Sequential(
            ResBlock(128, 64),
            ResBlock(64, 64),
        )

        deconv1 = nn.Sequential(
            ResBlock(64, 32),
            ResBlock(32, 32),
        )

        postprocess = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=(3, 3), padding=1),
            # nn.Tanh(),
            nn.Sigmoid()

        )

        # 3 w h

        self.device = device
        self.preprocess = preprocess.to(device)
        self.conv1 = conv1.to(device)
        self.conv2 = conv2.to(device)
        self.conv3 = conv3.to(device)
        # self.conv4 = conv4.to(device)
        self.res = res.to(device)
        self.deconv1 = deconv1.to(device)
        self.deconv2 = deconv2.to(device)
        self.deconv3 = deconv3.to(device)
        # self.deconv4 = deconv4.to(device)
        self.upsample1 = upsample1.to(device)
        self.upsample2 = upsample2.to(device)
        self.upsample3 = upsample3.to(device)
        # self.upsample4 = upsample4.to(device)
        self.postprocess = postprocess.to(device)

    def forward(self, x):
        x = x.to(self.device)

        l1 = self.preprocess(x)
        l2 = self.conv1(l1)
        l3 = self.conv2(l2)
        l4 = self.conv3(l3)
        # l5 = self.conv4(l4)

        # r5 = self.res(l5)
        r4 = self.res(l4)

        # r4p = self.upsample4(r5)
        # r4f = torch.cat([l4, r4p], dim=1)
        # r4 = self.deconv4(r4f)


        r3p = self.upsample3(r4)
        r3f = torch.cat([l3, r3p], dim=1)
        r3 = self.deconv3(r3f)
        r2p = self.upsample2(r3)
        r2f = torch.cat([l2, r2p], dim=1)
        r2 = self.deconv2(r2f)
        r1p = self.upsample1(r2)
        r1f = torch.cat([l1, r1p], dim=1)
        r1 = self.deconv1(r1f)

        output = self.postprocess(r1)
        # output = torch.nn.functional.interpolate(output, size=(224,224), mode='bilinear')

        return output




