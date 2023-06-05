import torch
from torch import nn
import torch.nn.functional as F
from model.resnet.resnet import Resnet

from torch.utils.data import DataLoader
from tools.dataloader import MyDatasets, shuffle, label_encoder
from tools import Mytransforms

import os
import json
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: [str, nn.Module], ori_img_size: [list, tuple] = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        # 原图尺寸
        self.ori_img_size = ori_img_size
        # 存放最后一层卷积层的输出
        self.feature_maps = None
        # 存放最后一层卷积层反向传播的梯度
        self.gradient = None
        # 注册相应的钩子
        self.hook_feature_map()
        self.hook_gradient()

    def hook_feature_map(self):
        # 获取指定层的输出结果
        def hook(module, input, output):
            self.feature_maps = (output.detach())

        if isinstance(self.target_layer, str):
            for name, module in self.model.named_modules():
                if name == self.target_layer:
                    # register_forward_hook被用来注册一个钩子函数，用于获取指定层的输出结果
                    # 将hook传入，将指定层输出传入self.feature_maps
                    module.register_forward_hook(hook)
        else:
            self.target_layer.register_forward_hook(hook)

    def hook_gradient(self):
        # 获取指定层的反向传播的梯度值
        def hook(module, grad_in, grad_out):
            self.gradient = grad_out[0].detach()

        if isinstance(self.target_layer, str):
            for name, module in self.model.named_modules():
                if name == self.target_layer:
                    # register_forward_hook被用来注册一个钩子函数，用于获取指定层的输出结果
                    # 将hook传入，将指定层输出传入self.feature_maps
                    module.register_full_backward_hook(hook)
        else:
            self.target_layer.register_full_backward_hook(hook)

    def forward(self, x):
        return self.model(x)

    def backward(self, y):
        y.backward(torch.ones_like(y))

    def __call__(self, x: torch.Tensor, y: torch.Tensor = None):
        # 进行前向传播和反向传播，期间会触发之前注册的钩子，自动将目标层输出和梯度赋给feature_maps和gradient
        self.forward(x)
        if y is None:
            y = self.forward(x)
        self.backward(y)

        # 计算grad_cam的权重
        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        # 双线性插值至原图大小
        if self.ori_img_size:
            cam = F.interpolate(cam, size=self.ori_img_size, mode='bilinear', align_corners=False)
        else:
            cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.cpu().numpy().transpose(0, 2, 3, 1)
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        # 分批次处理
        result = []
        for bs in range(x.size(0)):
            result.append(self.generate_gradcam_pic(x[bs], cam[bs]))

        return y, result

    def generate_gradcam_pic(self, img, grad_cam):
        # 啊啊啊啊啊啊啊啊！！！！写的好乱！！！！！！
        # 转换原图为RGB并恢复至resize前大小
        if self.ori_img_size:
            img = img.unsqueeze(dim=0)
            img = F.interpolate(img, size=self.ori_img_size, mode='bilinear', align_corners=False)
        img = (img * 255).squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # 将灰度图cam按数值生成热力图，仅将cam中不为0的部分转换成热力图
        mask = (grad_cam != 0).astype(np.uint8)
        heatmap = cv2.applyColorMap((grad_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # heatmap = heatmap * mask
        # 按热力图与原图融合并重新归一化
        result = 0.7 * img + 0.3 * heatmap
        result = (result / result.max() * 255).astype(np.uint8)

        return result


if __name__ == '__main__':
    data_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\data'
    data_path_txt = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\data\img_names.txt'
    cfg_file = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\model\config.json'
    pretrained_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\model\resnet\pretreatment_resnet34.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('The code will test in {}'.format(device))
    restore_scale = 1
    ori_img_size = [5472 * restore_scale, 3648 * restore_scale]

    labels = os.listdir(data_path)
    if 'img_names.txt' in labels:
        labels.remove('img_names.txt')

    # 在计算混淆矩阵时需传入参考标签的序号，防止输入的predict和labels不含有某一类别
    refer_labels = list(label_encoder(labels).values())

    # 划分数据集
    with open(data_path_txt, 'r', encoding='utf-8') as f:
        img_info = f.readlines()
    print("Successfully read img names from {}".format(data_path))

    # 打乱数据集
    # img_info = shuffle(img_info, 2)

    # 划分数据集
    data_num = len(img_info)
    train_data_info = img_info[:int(data_num * 0.7)]
    val_data_info = img_info[int(data_num * 0.7):int(data_num * 0.9)]
    test_data_info = img_info[int(data_num * 0.9):]
    print('train_data_num:', len(train_data_info))
    print('val_data_num:', len(val_data_info))
    print('test_data_num:', len(test_data_info))

    transformers = [
        Mytransforms.Resize((224, 224)),
        Mytransforms.ToTensor()
    ]

    test_datasets = MyDatasets(data_path, labels, test_data_info, transformers)
    test_dataloader = DataLoader(test_datasets, batch_size=2, shuffle=True)

    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    model = Resnet(cfg['resnet34'], 3, 2)

    model.load_state_dict(torch.load(pretrained_path, torch.device('cpu')))
    print('Successfully load pretrained model from {}'.format(pretrained_path))

    # 创建GradCAM，测试程序包含在内
    target_layer = model.bottleneck_4[-1]
    test_grad_cam = GradCAM(model, target_layer, ori_img_size)

    for i, (face_img, tongue_img, label) in enumerate(test_dataloader):
        tongue_img = tongue_img.to(device, dtype=torch.float)
        _, result = test_grad_cam(tongue_img)

        for j in range(len(result)):
            img = Image.fromarray(result[j]).convert('RGB')
            plt.imshow(img)
            plt.show()

        break

