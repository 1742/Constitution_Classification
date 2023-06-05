import sys

import torch
from torch import nn
from torchvision import models
from model.resnet.resnet import Conv2d

import os
import json


class SE_Resnet(nn.Module):
    def __init__(self, cfg: list, in_channels: int, num_classes: int = 1000):
        super(SE_Resnet, self).__init__()
        if len(cfg[0]) == 7:
            fc_cells = 2048
        else:
            fc_cells = 512
        # inplanes用于记录各basicblock输出，以便不同的bottleneck间连接
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.bottleneck_1 = self._make_bottleneck(cfg[0])
        self.bottleneck_2 = self._make_bottleneck(cfg[1])
        self.bottleneck_3 = self._make_bottleneck(cfg[2])
        self.bottleneck_4 = self._make_bottleneck(cfg[3])

        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(fc_cells, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.bottleneck_1(x)
        x = self.bottleneck_2(x)
        x = self.bottleneck_3(x)
        x = self.bottleneck_4(x)

        x = self.avepool(x)
        x = self.flatten(x)
        x = self.fc(x)

        # x = self.softmax(x)

        return x

    def _make_bottleneck(self, cfg: list):
        """
        生成Bottleneck

        :param cfg:
            各bottleneck的参数
        :return:
            nn.Sequential的bottleneck
        """
        if len(cfg) == 7:
            """
            resnet50、resnet101、resnet152模板
            """
            block = SE_l_BasicBlock
            in_channels, medium_channels, out_channels, kernel_size, stride, padding, layers_num = cfg
            down_sample = None
            if stride != 1 or in_channels != out_channels:
                down_sample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                    nn.BatchNorm2d(out_channels)
                )

            layers = []
            # 额外加入第一个basicblock，下采样basicblock一般在bottleneck第一位
            layers.append(block(self.inplanes, medium_channels, out_channels, kernel_size, stride, padding, down_sample))
            self.inplanes = out_channels
            for _ in range(layers_num - 1):
                layers.append(block(self.inplanes, medium_channels, out_channels))
        else:
            """
            resnet18、resnet34模板
            """
            block = SE_s_BasicBlock
            in_channels, out_channels, kernel_size, stride, padding, layers_num = cfg
            down_sample = None
            if stride != 1 or in_channels != out_channels:
                down_sample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                    nn.BatchNorm2d(out_channels)
                )

            layers = []
            # 额外加入第一个basicblock，下采样basicblock一般在bottleneck第一位
            layers.append(block(in_channels, out_channels, kernel_size, stride, padding, down_sample))
            self.inplanes = out_channels
            for _ in range(layers_num - 1):
                layers.append(block(self.inplanes, out_channels))

        return nn.Sequential(*layers)


class SE_l_BasicBlock(nn.Module):
    def __init__(self, in_channels: int, medium_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, down_sample=None, r: int = 16):
        super(SE_l_BasicBlock, self).__init__()
        self.down_sample = down_sample
        self.conv = nn.Sequential(
            Conv2d(in_channels, medium_channels, kernel_size=1, stride=1, padding=0),
            Conv2d(medium_channels, medium_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            Conv2d(medium_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

        self.relu = nn.ReLU(inplace=True)

        self.se_block = SE_Block(out_channels, r)

    def forward(self, x):
        res = x
        x = self.conv(x)

        x_se = self.se_block(x)
        x = x * x_se

        if self.down_sample is not None:
            res = self.down_sample(res)

        out = self.relu(x + res)

        return out


class SE_s_BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, down_sample=None, r: int = 16):
        super(SE_s_BasicBlock, self).__init__()
        self.down_sample = down_sample
        self.conv = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

        self.se_block = SE_Block(out_channels, r)

    def forward(self, x):
        res = x
        x = self.conv(x)

        x_se = self.se_block(x)
        x = x * x_se

        if self.down_sample is not None:
            res = self.down_sample(res)

        out = self.relu(x + res)

        return out


class SE_Block(nn.Module):
    def __init__(self, out_channels: int, r: int = 16):
        super(SE_Block, self).__init__()
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_channels, int(out_channels / r), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_channels / r), out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.se_block(x)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\model\config.json'
    pretrained_path = r'C:\Users\13632\.cache\torch\hub\checkpoints\resnet34-b627a593.pth'

    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    se_resnet = SE_Resnet(cfg['resnet34'], 3, 2)
    print(se_resnet)

    png = torch.randint(255, (1, 3, 224, 224)).float().to(device)
    out = se_resnet(png / 255.)
    print(out.size())


