import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleAutoEncoderMD(nn.Module):
    def __init__(self, size=(256, 512), resnet=None):
        super(SimpleAutoEncoderMD, self).__init__()
        self.size = size

        if not resnet:
            self.encoder = Encoder(resnet=resnet)
            self.decoder = Decoder(resnet=resnet)

    def forward(self, x):
        x = self.encoder(x)
        disp = self.decoder(x)
        return disp


class Encoder(nn.Module):
    def __init__(self, resnet=None):
        super(Encoder, self).__init__()

        if not resnet:
            # Input shape: [8, 3, 256, 512] == [batch, channel, height, width]
            c_in = 3            # [3  , 256, 512]
            c1_encoder = 64     # [64 , 128, 256]
            c2_encoder = 128    # [128,  64, 128]
            c3_encoder = 256    # [256,  32,  64]
            c_out = 512         # [512,  16,  32]
            self.encoder = nn.Sequential(
                nn.Conv2d(c_in, c1_encoder, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(c1_encoder, c2_encoder, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(c2_encoder, c3_encoder, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(c3_encoder, c_out, kernel_size=3, stride=2, padding=1),
            )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, resnet=None):
        super(Decoder, self).__init__()
        self.resnet = resnet
        if not resnet:
            self.c_in = 512          # [512,  16,  32]
            self.c1_decoder = 256    # [256,  32,  64] -> disp4
            self.c2_decoder = 128    # [128,  64, 128] -> disp3
            self.c3_decoder = 256    # [64 , 128, 256] -> disp2
            self.c_out = 32          # [32 , 256, 512] -> disp1
            self.upConv1 = nn.ConvTranspose2d(self.c_in, self.c1_decoder, kernel_size=4, stride=2, padding=1)
            self.upConv2 = nn.ConvTranspose2d(self.c1_decoder, self.c2_decoder, kernel_size=4, stride=2, padding=1)
            self.upConv3 = nn.ConvTranspose2d(self.c2_decoder, self.c3_decoder, kernel_size=4, stride=2, padding=1)
            self.upConv4 = nn.ConvTranspose2d(self.c3_decoder, self.c_out, kernel_size=4, stride=2, padding=1)
            self.ELU = nn.ELU()

    def forward(self, x):
        if not self.resnet:
            x = self.upConv1(x)
            x = self.ELU(x)
            disp4 = GetDisp(self.c1_decoder)(x)
            x = self.upConv2(x)
            x = self.ELU(x)
            disp3 = GetDisp(self.c2_decoder)(x)
            x = self.upConv3(x)
            x = self.ELU(x)
            disp2 = GetDisp(self.c3_decoder)(x)
            x = self.upConv4(x)
            x = self.ELU(x)
            disp1 = GetDisp(self.c_out)(x)

            return disp1, disp2, disp3, disp4


class GetDisp(nn.Module):
    def __init__(self, num_input_layers):
        """
        Class to get disparity map at different scale. Referenced from 3.2. Depth Estimation Network
        :param num_input_layers: number of layers of the current deconvolution layer
        """""
        super(GetDisp, self).__init__()
        self.conv = nn.Conv2d(num_input_layers, 2, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(2)
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        padding = (p, p, p, p)
        x = self.conv(F.pad(x, padding))
        x = self.normalize(x)
        return 0.3 * self.activate(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        padding = (p, p, p, p)
        x = self.conv(F.pad(x, padding))
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv(x)


class MaxPool(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        padding = (p, p, p, p)
        x = F.pad(x, padding)
        x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=2)
        return x


class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResConv, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=stride)
        self.conv3 = nn.Conv2d(out_channels, 4*out_channels, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels, 4*out_channels, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4*out_channels)

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        shortcut = self.conv4(x)
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


def res_block(in_channels, out_channels, num_blocks, stride):
    layers = list()
    layers.append(ResConv(in_channels, out_channels, stride=stride))
    for _ in range(1, num_blocks - 1):
        layers.append(ResConv(4 * out_channels, out_channels, 1))
    layers.append(ResConv(4 * out_channels, out_channels, 1))
    return nn.Sequential(*layers)


class Resnet50MonoDepth(nn.Module):
    def __init__(self, in_channels):
        super(Resnet50MonoDepth, self).__init__()
        # Encoder
        self.conv1 = Conv(in_channels, 64, kernel_size=7, stride=2)
        self.pool1 = MaxPool(kernel_size=3)
        self.conv2 = res_block(64, 64, 3, stride=2)
        self.conv3 = res_block(256, 128, 4, stride=2)
        self.conv4 = res_block(512, 256, 6, stride=2)
        self.conv5 = res_block(1024, 512, 3, stride=2)
        # Decoder
        self.upconv6 = UpConv(2048, 512, 3, 2)
        self.iconv6 = Conv(1024 + 512, 512, 3, 1)

        self.upconv5 = UpConv(512, 256, 3, 2)
        self.iconv5 = Conv(512 + 256, 256, 3, 1)

        self.upconv4 = UpConv(256, 128, 3, 2)
        self.iconv4 = Conv(256 + 128, 128, 3, 1)
        self.disp4_layer = GetDisp(128)

        self.upconv3 = UpConv(128, 64, 3, 2)
        self.iconv3 = Conv(64 + 64 + 2, 64, 3, 1)
        self.disp3_layer = GetDisp(64)

        self.upconv2 = UpConv(64, 32, 3, 2)
        self.iconv2 = Conv(32 + 64 + 2, 32, 3, 1)
        self.disp2_layer = GetDisp(32)

        self.upconv1 = UpConv(32, 16, 3, 2)
        self.iconv1 = Conv(16 + 2, 16, 3, 1)
        self.disp1_layer = GetDisp(16)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4

        # decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        disp4 = self.disp4_layer(iconv4)
        udisp4 = nn.functional.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, udisp4), 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3_layer(iconv3)
        udisp3 = nn.functional.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, udisp3), 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2_layer(iconv2)
        udisp2 = nn.functional.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, udisp2), 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1_layer(iconv1)

        return disp1, disp2, disp3, disp4


if __name__ == '__main__':
    tesT_input = torch.randn(8, 3, 256, 512)
    # test_model = Conv(3, 64, kernel_size=3, stride=2)
    # test_model = MaxPool(3)
    test_model = Resnet50MonoDepth(3)
    output = test_model(tesT_input)
    for disp in output:
        print(disp.shape)
