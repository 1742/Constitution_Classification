import torch
from torch import nn
from torchvision import models
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


class resnet_l_BasicBlock(nn.Module):
    def __init__(self, in_channels: int, medium_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, down_sample=None):
        super(resnet_l_BasicBlock, self).__init__()
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


class resnet_s_BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, down_sample=None):
        super(resnet_s_BasicBlock, self).__init__()
        self.down_sample = down_sample
        self.conv = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
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
    def __init__(self, cfg: list, in_channels: int, num_classes: int = 1000):
        super(Resnet, self).__init__()
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

        self.avepool = nn.AvgPool2d(7)
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
            block = resnet_l_BasicBlock
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
            block = resnet_s_BasicBlock
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


class pretrained_Resnet(object):
    def __init__(self, model_name: str, device: str, in_channels: int, num_classes: int = 1000, pretrained_path: str = None):
        super(pretrained_Resnet).__init__()
        ori_model_dir = r'C:\Users\13632\.cache\torch\hub'
        if pretrained_path:
            if model_name == 'Resnet50' or model_name == 'resnet50':
                model = models.resnet50()
            elif model_name == 'Resnet34' or model_name == 'resnet34':
                model = models.resnet34()
            elif model_name == 'Resnet18' or model_name == 'resnet18':
                model = models.resnet18()
            model.load_state_dict(torch.load(pretrained_path, torch.device(device)))
        else:
            if model_name == 'Resnet50' or model_name == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            elif model_name == 'Resnet34' or model_name == 'resnet34':
                model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            elif model_name == 'Resnet18' or model_name == 'resnet18':
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        print('Successfully loaded pretrained weights.')

        # 更改模型的输入和输出
        model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        self.features = model

        if num_classes == 1:
            self.classifier = nn.Sigmoid()
        else:
            self.classifier = nn.Softmax(dim=1)

    def __call__(self):
        return nn.Sequential(self.features), nn.Sequential(self.classifier)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\model\config.json'
    # pretrained_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\model\checkpoints\resnet34-b627a593.pth'

    if not os.path.exists(cfg_path):
        cfg = {
            "vgg16": [
                [3, 64, 2],
                [64, 128, 2],
                [128, 256, 2],
                [256, 512, 3],
                [512, 512, 3]
            ],
            'resnet18': [
                [64, 64, 3, 1, 1, 2],
                [64, 128, 3, 2, 1, 2],
                [128, 256, 3, 2, 1, 2],
                [256, 512, 3, 2, 1, 2],
            ],
            'resnet34': [
                [64, 64, 3, 1, 1, 3],
                [64, 128, 3, 2, 1, 4],
                [128, 256, 3, 2, 1, 6],
                [256, 512, 3, 2, 1, 3],
            ],
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

    # model = Resnet(cfg['resnet18'], 3, 2)
    # print(model)
    # png = torch.randint(255, (1, 3, 224, 224)).float().to('cpu')
    # out = model(png / 255.)
    # print(out.size())

    resnet = pretrained_Resnet('resnet50', device, 3, 2)
    feature, classifier = resnet()
    print(feature)
    print(classifier)

    # png = torch.randint(255, (1, 3, 224, 224)).float().to('cpu')
    # out = feature(png / 255.)
    # print(out.size())

