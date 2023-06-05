import sys

import torch
from torch import nn
import torch.nn.functional as F
from model.resnet.resnet import Resnet

from torch.utils.data import Dataset, DataLoader
from tools.dataloader import MyDatasets, shuffle, label_encoder
from torchvision import transforms
from tools import Mytransforms
import numpy as np

import os
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.evaluation_index import Accuracy, Confusion_matrix, ROC_and_AUC, plot_ROC, Visualization
from tools.grad_cam import GradCAM


data_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\data'
data_path_txt = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\data\img_names.txt'
cfg_file = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\model\config.json'
pretrained_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\model\resnet\pretreatment_resnet34.pth'
save_gradcam_picture_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\runs\pretreatment_resnet34\test\grad_cam'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('The predict will run in {} ...'.format(device))
pretrained = True
save_option = True
restore_scale = 2


def generate_gradcam(
        device: str,
        model: nn.Module,
        target_layer: [str, nn.Module],
        ori_img_size: [list, tuple],
        test_datasets: MyDatasets,
        refer_labels: dict,
        pretrained_path: str,
        save_option: bool,
):
    # 加载权重
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, torch.device(device)))
        print('Successfully load pretrained model from {}'.format(pretrained_path))
    else:
        print('model parameters files is not exist!')
        sys.exit(0)
    model.to(device)

    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False)

    # 实例化GradCAM
    gradcam = GradCAM(model, target_layer, ori_img_size)

    model.eval()
    with tqdm(total=len(test_dataloader)) as pbar:
        pbar.set_description('loading')
        for i, (face_img, tongue_img, label) in enumerate(test_dataloader):
            # face_img = face_img.to(device, dtype=torch.float)
            tongue_img = tongue_img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)

            output = model(tongue_img).float()
            pred = torch.softmax(output, dim=1).argmax(dim=1)

            pbar.update(1)

            if save_option:
                # 保存grad_cam图
                _, grad_cam_pic = gradcam(tongue_img)
                for cam in grad_cam_pic:
                    # 转换tongue_img和cam为图片
                    tongue_img = Image.fromarray(np.array((tongue_img * 255).squeeze().permute(1, 2, 0), dtype=np.uint8)).convert('RGB')
                    cam = Image.fromarray(cam).convert('RGB')

                    # 拼接原图和cam
                    size = cam.size
                    result = Image.new('RGB', (size[0] * 2, size[1]), 0)
                    result.paste(tongue_img, (0, 0))
                    result.paste(cam, (size[0] + 1, 0))

                    # 打上指标
                    plt.imshow(result)
                    plt.title('pred: {}  label: {}'.format(refer_labels[int(pred)], refer_labels[int(label)]))
                    plt.axis('off')
                    plt.savefig(save_gradcam_picture_path + '\\{}.png'.format(i))

    print('Successfully save grad-cam picture in {}'.format(save_gradcam_picture_path))


if __name__ == '__main__':
    labels = os.listdir(data_path)
    if 'img_names.txt' in labels:
        labels.remove('img_names.txt')

    # 用于在生成图片时打上标签
    refer_labels = dict()
    labels = label_encoder(labels)
    items = labels.items()
    for value, key in items:
        refer_labels[key] = value

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

    test_datasets = MyDatasets(data_path, labels, test_data_info[:10], transformers)

    # 原图尺寸及高清回复倍率
    ori_img_size = [int(5472 * restore_scale), int(3648 * restore_scale)]
    print('origin image size(included restore):', ori_img_size)

    # 读取模型结构
    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    model = Resnet(cfg['resnet34'], 3, 2)

    # gradcam目标层
    target_layer = model.bottleneck_4[-1].conv[-2]
    print('target layer:', target_layer)

    generate_gradcam(
        device=device,
        model=model,
        test_datasets=test_datasets,
        refer_labels=refer_labels,
        target_layer=target_layer,
        ori_img_size=None,
        pretrained_path=pretrained_path,
        save_option=save_option,
    )
