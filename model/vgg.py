import torch
from torch import nn
import os
import json


class VGG16(nn.Module):
    def __init__(self, cfg: list, cls_num: int):
        super(VGG16, self).__init__()
        self.cfg = cfg

        self.layers_1 = self._make_layers(cfg[0])
        self.layers_2 = self._make_layers(cfg[1])
        self.layers_3 = self._make_layers(cfg[2])
        self.layers_4 = self._make_layers(cfg[3])
        self.layers_5 = self._make_layers(cfg[4])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

        self.fc6 = nn.Linear(25088, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, cls_num)

    def forward(self, x):
        x = self.maxpool(self.layers_1(x))
        x = self.maxpool(self.layers_2(x))
        x = self.maxpool(self.layers_3(x))
        x = self.maxpool(self.layers_4(x))
        x = self.maxpool(self.layers_5(x))

        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc6(x)))
        x = self.dropout(self.relu(self.fc7(x)))
        x = self.fc8(x)

        return x

    def _make_layers(self, params: list):
        layers = []

        in_channel = params[0]
        out_channel = params[1]
        for _ in range(params[2]):
            layers.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            in_channel = out_channel

        return nn.Sequential(*layers)


class Multiple_Image_in_Feature_VGG16(nn.Module):
    def __init__(self, cfg: list, cls_num: int):
        super(Multiple_Image_in_Feature_VGG16, self).__init__()
        self.cfg = cfg

        self.layers_1_1 = self._make_layers(cfg[0])
        self.layers_1_2 = self._make_layers(cfg[1])
        self.layers_1_3 = self._make_layers(cfg[2])
        self.layers_1_4 = self._make_layers(cfg[3])
        self.layers_1_5 = self._make_layers(cfg[4])

        self.layers_2_1 = self._make_layers(cfg[0])
        self.layers_2_2 = self._make_layers(cfg[1])
        self.layers_2_3 = self._make_layers(cfg[2])
        self.layers_2_4 = self._make_layers(cfg[3])
        self.layers_2_5 = self._make_layers(cfg[4])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

        self.fc6 = nn.Linear(25088 * 2, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.fc8 = nn.Linear(4096, cls_num)

    def forward(self, x, y):
        x = self.maxpool(self.layers_1_1(x))
        x = self.maxpool(self.layers_1_2(x))
        x = self.maxpool(self.layers_1_3(x))
        x = self.maxpool(self.layers_1_4(x))
        x = self.maxpool(self.layers_1_5(x))

        x = self.flatten(x)

        y = self.maxpool(self.layers_2_1(y))
        y = self.maxpool(self.layers_2_2(y))
        y = self.maxpool(self.layers_2_3(y))
        y = self.maxpool(self.layers_2_4(y))
        y = self.maxpool(self.layers_2_5(y))

        y = self.flatten(y)

        z = self.dropout(self.relu(self.fc6(torch.cat((x, y), dim=1))))
        z = self.dropout(self.relu(self.fc7(z)))

        z = self.fc8(z)

        return z

    def _make_layers(self, params: list):
        layers = []

        in_channel = params[0]
        out_channel = params[1]
        for _ in range(params[2]):
            layers.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            in_channel = out_channel

        return nn.Sequential(*layers)


class Multiple_Image_in_Decision_VGG16(nn.Module):
    def __init__(self, cfg: list, cls_num: int):
        super(Multiple_Image_in_Decision_VGG16, self).__init__()
        self.vgg_1 = VGG16(cfg, cls_num)
        self.vgg_2 = VGG16(cfg, cls_num)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        x = self.vgg_1(x)
        y = self.vgg_2(y)

        pred = self.softmax(x + y)

        return pred


def get_parameter_number(model: nn.Module):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    cfg_path = './config.json'
    if not os.path.exists(cfg_path):
        cfg = [
            [3, 64, 2],
            [64, 128, 2],
            [128, 256, 2],
            [256, 512, 3],
            [512, 512, 3],
        ]
        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'vgg16': cfg}))

    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # model = VGG16(cfg['vgg16'], 3)
    # model = Multiple_Image_in_Feature_VGG16(cfg['vgg16'], 3)
    model = Multiple_Image_in_Decision_VGG16(cfg['vgg16'], 3)
    print(model)
    params_info = get_parameter_number(model)
    print('Total params:', params_info['Total'])
    print('trainable params:', params_info['Trainable'])
