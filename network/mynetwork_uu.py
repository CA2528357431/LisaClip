import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化高宽为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化高宽为1

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化---》1*1卷积层降维----》激活函数----》卷积层升维
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化---》1*1卷积层降维----》激活函数----》卷积层升维
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out  # 加和操作
        return self.sigmoid(out)  # sigmoid激活操作


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        # 经过一个卷积层，输入维度是2，输出维度是1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # sigmoid激活操作

        self.hot = None

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的平均值  b,1,h,w
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的最大值  b,1,h,w
        x = torch.cat([avg_out, max_out], dim=1)  # 在第一维度上拼接，变为 b,2,h,w
        x = self.conv1(x)  # 转换为维度，变为 b,1,h,w
        self.hot = self.sigmoid(x)  # sigmoid激活操作
        return self.hot


class cbamblock(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(cbamblock, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)  # 将这个权值乘上原输入特征层
        x = x * self.spatialattention(x)  # 将这个权值乘上原输入特征层
        return x




class NoiseInjection(nn.Module):
    def __init__(self, weight):
        super().__init__()

        self.weight = weight*5

    def forward(self, image):

        return image

        batch, channel, height, width = image.shape
        noise = image.new_empty(batch, channel, height, width).normal_()
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
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1)),
            # nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        block2 = nn.Sequential(
            # nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=2, output_padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
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
        # self.noise = NoiseInjection()


    def forward(self, x):
        x = self.block1(x) + self.skip(x)
        res = self.block2(x)
        # res = self.noise(res)
        return res


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        block = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1)),
            # cbamblock(out_channel),
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

class ResBlock_attention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        block = nn.Sequential(
            # nn.ReflectionPad2d(1),
            cbamblock(channel),
            nn.InstanceNorm2d(channel),
            nn.ReLU(),
        )
        self.block = block
        self.in_channel = channel
        self.out_channel = channel
#         self.noise = NoiseInjection()

    def forward(self, x):
        return self.block(x) + x


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
            # ResBlock_attention(32),
            ResBlock(32, 32),
            # NoiseInjection(0.2),
            # cbamblock(32)
        )
        # 32 w h
        conv1 = nn.Sequential(
            DownSample(32, 64),
            # ResBlock(64, 64),
            NoiseInjection(0.02),

        )
        # 64 w/2 h/2
        conv2 = nn.Sequential(
            DownSample(64, 128),
            # ResBlock(128, 128),
            NoiseInjection(0.02),
        )
        # 128 w/4 h/4
        conv3 = nn.Sequential(
            DownSample(128, 256),
            # ResBlock(256, 256),
            NoiseInjection(0.04),
        )
        # 256 w/8 h/8
        # conv4 = nn.Sequential(
        #     DownSample(256, 512),
        #     # ResBlock(512, 512),
        # )
        # 512 w/16 h/16

        res = nn.Sequential(
            ResBlock_attention(256),
            # ResBlock(256, 256),
            # cbamblock(256),
            # ResBlock(256, 256),
            NoiseInjection(0.04),

            ResBlock_attention(256),
            # ResBlock(256, 256),
            # cbamblock(256),
            # ResBlock(256, 256),
            NoiseInjection(0.04),
        )
        res2 = nn.Sequential(
            ResBlock_attention(128),
            # ResBlock(128, 128),
            # cbamblock(128),
            # ResBlock(128, 128),
            NoiseInjection(0.02),
            ResBlock_attention(128),
            # ResBlock(128, 128),
            # cbamblock(128),
            # ResBlock(128, 128),
            NoiseInjection(0.02),
        )
        # 512 w/16 h/16

        # upsample4 = nn.Sequential(
        #     UpSample(512, 256),
        #     # ResBlock(256, 256),
        # )

        upsample3 = nn.Sequential(
            UpSample(256, 128),
            # ResBlock(128, 128),
            # NoiseInjection(0.5),
        )

        upsample2 = nn.Sequential(
            UpSample(128, 64),
            # ResBlock(64, 64),
            # NoiseInjection(0.4),
        )

        upsample1 = nn.Sequential(
            UpSample(64, 32),
            # ResBlock(32, 32),
            # NoiseInjection(0.3),
        )

        # deconv4 = nn.Sequential(
        #     ResBlock(512, 256),
        #     ResBlock(256, 256),
        # )

        deconv3 = nn.Sequential(
            ResBlock(256, 128),
            NoiseInjection(0.04),
            # ResBlock(128, 128),
            # NoiseInjection(0.02),
        )

        deconv2 = nn.Sequential(
            ResBlock(128, 64),
            NoiseInjection(0.02),
            # ResBlock(64, 64),
            # NoiseInjection(0.02),
        )

        deconv1 = nn.Sequential(
            ResBlock(64, 32),
            NoiseInjection(0.02),
            # ResBlock(32, 32),
            # NoiseInjection(0.02),
        )

        postprocess = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=(3, 3), padding=(1, 1)),
            # nn.Tanh(),
            nn.Sigmoid(),

        )

        # 3 w h

        self.device = device
        self.preprocess = preprocess.to(device)
        self.conv1 = conv1.to(device)
        self.conv2 = conv2.to(device)
        self.conv3 = conv3.to(device)
        # self.conv4 = conv4.to(device)
        self.res = res.to(device)
        self.res2 = res2.to(device)
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
        r3f = torch.cat([self.res2(l3), r3p], dim=1)
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