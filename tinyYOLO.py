import torch
import torch.nn as nn


class YOLOConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(YOLOConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.nonlinear = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.nonlinear(self.bn(self.conv(x)))


class TinyDarkNet(nn.Module):
    def __init__(self):
        super(TinyDarkNet, self).__init__()

        self.net = nn.Sequential(
            YOLOConv2d(3, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            YOLOConv2d(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            YOLOConv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            YOLOConv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            YOLOConv2d(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            YOLOConv2d(256, 512, 3, 1, 1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(2, 1),

            YOLOConv2d(512, 1024, 3, 1, 1),
        )
    def forward(self, x):
        return self.net(x)

class TinyYOLO(nn.Module):
    def __init__(self):
        super(TinyYOLO, self).__init__()

        self.darknet = TinyDarkNet()
        self.NOC = nn.Sequential(
            YOLOConv2d(1024, 1024, 3, 1, 1),
            nn.Conv2d(1024, 30, 1, 1, 0)
        )

    def forward(self, x):
        feature = self.darknet(x)
        return self.NOC(feature)
