import torch
import torch.nn as nn

# Basic CNN：經典 CNN（卷積 + max pooling + 全連接層）


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # 輸入：(B, 3, 32, 32)
        # 輸出：(B, 32, 32, 32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # 輸出：(B, 32, 16, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 輸出：(B, 64, 16, 16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # 輸出：(B, 64, 8, 8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 輸出：(B, 128, 8, 8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        # 輸出：(B, 128*8*8) = (B, 8192)
        self.flatten = nn.Flatten()
        # 輸出：(B, 512)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu4 = nn.ReLU()
        # 輸出：(B, 10)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# Plain CNN：對照組，結構與 ResNet-20 相同，只拿掉 skip connection
# 為了維持相同結構，這裡用 stride=2 的卷積降採樣，不用 pooling


class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        return out


class PlainCNN(nn.Module):
    def __init__(self):
        super(PlainCNN, self).__init__()
        # 輸出：(B, 16, 32, 32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 輸出：(B, 16, 32, 32)
        self.layer1 = nn.Sequential(
            PlainBlock(16, 16), PlainBlock(16, 16), PlainBlock(16, 16))
        # 輸出：(B, 32, 16, 16)
        self.layer2 = nn.Sequential(
            PlainBlock(16, 32, stride=2), PlainBlock(32, 32), PlainBlock(32, 32))
        # 輸出：(B, 64, 8, 8)
        self.layer3 = nn.Sequential(
            PlainBlock(32, 64, stride=2), PlainBlock(64, 64), PlainBlock(64, 64))

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
