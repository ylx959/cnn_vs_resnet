import torch
import torch.nn as nn
import torch.nn.functional as F

# ResNet-20：CIFAR 版 ResNet（He et al. 2015 第 4.2 節）


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.stride = stride
        self.extra_channels = out_channels - in_channels

    def shortcut(self, x):
        # 尺寸不變時直接相加
        if self.stride == 1 and self.extra_channels == 0:
            return x

        # 尺寸改變時用 option A：隔點取樣縮小，通道不足的部分補 0
        # 這個做法沒有參數，所以參數量與 PlainCNN 完全相同
        x = x[:, :, ::self.stride, ::self.stride]
        half = self.extra_channels // 2
        return F.pad(x, (0, 0, 0, 0, half, half))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = torch.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self):
        super(ResNet20, self).__init__()
        # 輸出：(B, 16, 32, 32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 輸出：(B, 16, 32, 32)
        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16), ResidualBlock(16, 16), ResidualBlock(16, 16))
        # 輸出：(B, 32, 16, 16)
        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2), ResidualBlock(32, 32), ResidualBlock(32, 32))
        # 輸出：(B, 64, 8, 8)
        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64, stride=2), ResidualBlock(64, 64), ResidualBlock(64, 64))

        # 輸出：(B, 64, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 輸出：(B, 10)
        self.fc = nn.Linear(64, 10)

        # 卷積層使用 Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
