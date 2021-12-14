import torch
import torch.nn as nn
import torch.nn.functional as F

# our_in_ch = 1
# our_out_ch = 1

# ---------------------------------------------------------------------------- #
# Unet
# ---------------------------------------------------------------------------- #
# class conv_block(nn.Module):
#     """
#     Convolution Block
#     """
#
#     def __init__(self, in_ch, out_ch):
#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
# class up_conv(nn.Module):
#     """
#     Up Convolution Block
#     """
#
#     def __init__(self, in_ch, out_ch):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.up(x)
#         return x
#
#
# class U_Net(nn.Module):
#     """
#     UNet - Basic Implementation
#     Paper : https://arxiv.org/abs/1505.04597
#     """
#
#     def __init__(self, in_ch=1, out_ch=2):
#         super(U_Net, self).__init__()
#         print("Using U_Net...")
#         # in_ch =64, out_ch= ?
#         filters = [64, 128, 256, 512, 1024]
#
#         self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.Conv1 = conv_block(in_ch, filters[0])
#         self.Conv2 = conv_block(filters[0], filters[1])
#         self.Conv3 = conv_block(filters[1], filters[2])
#         self.Conv4 = conv_block(filters[2], filters[3])
#         self.Conv5 = conv_block(filters[3], filters[4])
#
#         self.Up5 = up_conv(filters[4], filters[3])
#         self.Up_conv5 = conv_block(filters[4], filters[3])
#
#         self.Up4 = up_conv(filters[3], filters[2])
#         self.Up_conv4 = conv_block(filters[3], filters[2])
#
#         self.Up3 = up_conv(filters[2], filters[1])
#         self.Up_conv3 = conv_block(filters[2], filters[1])
#
#         self.Up2 = up_conv(filters[1], filters[0])
#         self.Up_conv2 = conv_block(filters[1], filters[0])
#
#         self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
#
#         self.active = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         e1 = self.Conv1(x)
#         e2 = self.Maxpool1(e1)
#         e2 = self.Conv2(e2)
#         e3 = self.Maxpool2(e2)
#         e3 = self.Conv3(e3)
#         e4 = self.Maxpool3(e3)
#         e4 = self.Conv4(e4)
#         e5 = self.Maxpool4(e4)
#         e5 = self.Conv5(e5)
#         d5 = self.Up5(e5)
#         d5 = torch.cat((e4, d5), dim=1)
#         d5 = self.Up_conv5(d5)
#         d4 = self.Up4(d5)
#         d4 = torch.cat((e3, d4), dim=1)
#         d4 = self.Up_conv4(d4)
#         d3 = self.Up3(d4)
#         d3 = torch.cat((e2, d3), dim=1)
#         d3 = self.Up_conv3(d3)
#         d2 = self.Up2(d3)
#         d2 = torch.cat((e1, d2), dim=1)
#         d2 = self.Up_conv2(d2)
#         d1 = self.Conv(d2)
#         # print('output channel shape', out.size())
#         out = self.active(d1)
#
#         return out


###############################
# Unet from carnava
##############################

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
