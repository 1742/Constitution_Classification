import torch
from torch import nn
from torch.hub import load_state_dict_from_url
import os
import json


class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, medium_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, down_sample=None):
        super(BasicBlock, self).__init__()
        self.down_sample = down_sample
        self.conv = nn.Sequential(
            Conv2d(in_channels, medium_channels, kernel_size=1, stride=1, padding=0),
            Conv2d(medium_channels, medium_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            Conv2d(medium_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.conv(x)

        if self.down_sample is not None:
            res = self.down_sample(res)

        out = self.relu(x + res)

        return out


class Resnet(nn.Module):
    def __init__(self, cfg: list, in_channels: int, block: nn.Module = BasicBlock, num_classes: int = 1000):
        super(Resnet, self).__init__()
        # inplanes用于记录各basicblock输出，以便不同的bottleneck间连接
        self.inplanes = 64
        self.block = block
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.bottleneck_1 = self._make_bottleneck(cfg[0])
        self.bottleneck_2 = self._make_bottleneck(cfg[1])
        self.bottleneck_3 = self._make_bottleneck(cfg[2])
        self.bottleneck_4 = self._make_bottleneck(cfg[3])

        self.avepool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, num_classes)

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

        x = self.softmax(x)

        return x

    def _make_bottleneck(self, cfg: list):
        """
        生成Bottleneck

        :param cfg:
            各bottleneck的参数
        :return:
            nn.Sequential的bottleneck
        """
        in_channels, medium_channels, out_channels, kernel_size, stride, padding, layers_num = cfg
        down_sample = None
        if stride != 1 or in_channels != out_channels:
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        # 额外加入第一个basicblock，下采样basicblock一般在bottleneck第一位
        layers.append(self.block(self.inplanes, medium_channels, out_channels, kernel_size, stride, padding, down_sample))
        self.inplanes = out_channels
        for _ in range(layers_num - 1):
            layers.append(self.block(self.inplanes, medium_channels, out_channels))

        return nn.Sequential(*layers)


class Resnet50(object):
    def __init__(self, cfg: list, in_channels: int, block: nn.Module = BasicBlock, num_classes: int = 1000, pretrained=False):
        super(Resnet50).__init__()
        model = Resnet(cfg, in_channels, block, num_classes)
        if pretrained:
            state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth",
                                                  model_dir="./model_data")
            model.load_state_dict(state_dict)

        self.features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.bottleneck_1,
                              model.bottleneck_2, model.bottleneck_3])
        self.classifier = list([model.bottleneck_4, model.avepool])

    def __call__(self):
        return nn.Sequential(*self.features), nn.Sequential(*self.classifier)


if __name__ == '__main__':
    cfg_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Faster-RCNN\model\config.json'
    if not os.path.exists(cfg_path):
        cfg = {
            'resnet50': [
                [64, 64, 256, 3, 1, 1, 3],
                [256, 128, 512, 3, 2, 1, 4],
                [512, 256, 1024, 3, 2, 1, 6],
                [1024, 512, 2048, 3, 2, 1, 3],
            ]
        }
        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(cfg))

    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    model = Resnet(cfg['resnet50'], 3, BasicBlock, 1)
    print(model)
    png = torch.randint(255, (1, 3, 224, 224)).float().to('cpu')
    out = model(png / 255.)
    print(out.size())

    # model = Resnet50(cfg['resnet50'], 3, BasicBlock, 2)
    # feature, classifier = model()
    # print(feature)
    # print(classifier)

    # png = torch.randint(255, (1, 3, 224, 224)).float().to('cpu')
    # out = feature(png / 255.)
    # print(out.size())

