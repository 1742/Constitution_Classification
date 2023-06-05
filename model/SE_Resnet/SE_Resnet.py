import sys

import torch
from torch import nn
from torchvision import models
from model.resnet.resnet import pretrained_Resnet

import os


class SE_Resnet(nn.Module):
    def __init__(self, model_name: str, device: str, in_channels: int, num_classes: int = 1000, pretrained_path: str = None):
        super(SE_Resnet, self).__init__()
        if pretrained_path:
            model = pretrained_Resnet(model_name, device, in_channels, num_classes, pretrained_path)
            model = model(False)
        else:
            if model_name == 'Resnet50' or model_name == 'resnet50':
                model = models.resnet50()
            elif model_name == 'Resnet34' or model_name == 'resnet34':
                model = models.resnet34()
            elif model_name == 'Resnet18' or model_name == 'resnet18':
                model = models.resnet18()

            # 更改模型的输入和输出
            model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)

        # 拆分封装好的模型
        children = list(model.children())
        self.input = nn.Sequential(*children[:4])
        self.features = children[4:-2]

        if num_classes == 1:
            self.classifier = nn.Sequential(
                children[-2],
                nn.Flatten(),
                children[-1],
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                children[-2],
                nn.Flatten(),
                children[-1],
                nn.Softmax(dim=1)
            )

        self.se_blocks = self._make_se_net(children)

    def forward(self, x):
        x = self.input(x)

        for basicblock, seblock in zip(self.features, self.se_blocks):
            x = basicblock(x)
            x_se = seblock(x)
            x = x * x_se

        x = self.classifier(x)

        return x

    def _make_se_net(self, children: list):
        se_blocks = []
        for bottleneck in children[4:-2]:
            out_channels = bottleneck[-1].bn2.num_features
            se_blocks.append(self._se_block(out_channels))

        return se_blocks

    def _se_block(self, out_channels: int, r: int = 16):
        se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_channels, int(out_channels / r), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_channels / r), out_channels, 1),
            nn.Sigmoid()
        )

        return se_block


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_path = r'C:\Users\13632\.cache\torch\hub\checkpoints\resnet34-b627a593.pth'

    se_resnet = SE_Resnet('resnet34', device, 3, 2, pretrained_path)

    png = torch.randint(255, (1, 3, 224, 224)).float().to(device)
    out = se_resnet(png / 255.)
    print(out.size())


