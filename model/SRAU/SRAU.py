import torch
from torch import nn
import torch.nn.functional as F
from model.SE_Resnet.SE_Resnet import SE_Resnet

import json
import sys


# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        # 池化 -> 1*1 卷积 -> 上采样
        # 继承的是nn.Sequential。。。
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels: int = 256, atrous_rates: list = [6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        )

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)

        return self.project(res)


class SRAU(nn.Module):
    def __init__(self, cfg: list, in_channels: int, num_classess: int, dilation: list = [6, 12, 18]):
        super(SRAU, self).__init__()
        model = SE_Resnet(cfg, in_channels, num_classess)
        modules = list(model.children())[:-4]

        aspp = ASPP(cfg[1][1], cfg[1][1], dilation)

        modules.insert(6, aspp)
        self.moudules = nn.Sequential(*modules)

        self.dropout = nn.Dropout(0.2)

        num_classess = 1 if num_classess == 2 else num_classess
        if num_classess == 1:
            self.out_pred = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(cfg[-1][1], num_classess),
                nn.Sigmoid()
            )
            self.out_data_uncertainty = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(cfg[-1][1], num_classess),
                nn.Sigmoid()
            )
        else:
            self.out_pred = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(cfg[-1][1], num_classess),
                nn.Softmax(dim=1)
            )
            self.out_data_uncertainty = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(cfg[-1][1], num_classess),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.moudules(x)
        x = self.dropout(x)
        x_pred = self.out_pred(x)
        x_data_uncertainty = self.out_data_uncertainty(x)

        return x_pred.float(), x_data_uncertainty.float()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_file = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\model\config.json'

    # 读取模型结构
    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    model = SRAU(cfg['resnet34'], 3, 2)
    print(model)

    png = torch.randint(255, (1, 3, 224, 224)).float().to(device)
    pred, data_uncertainty = model(png / 255.)
    print(pred)
    print(data_uncertainty)

