import sys

import torch
from torch import nn
from model.resnet.resnet import Resnet, BasicBlock

from torch.utils.data import Dataset, DataLoader
from tools.dataloader import MyDatasets, shuffle, label_encoder
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tools import Mytransforms
import numpy as np

import os
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.evaluation_index import Accuracy, Confusion_matrix, ROC_and_AUC, plot_ROC, Visualization


data_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\data'
data_path_txt = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\data\img_names.txt'
cfg_file = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\model\config.json'
indicator_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\runs'
pretrained_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\model\resnet\resnet50.pth'
effect_path = r'runs\resnet50\result'
save_predict_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\runs\resnet50\result\result.txt'


batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('The predict will run in {} ...'.format(device))
pretrained = True
save_option = True


def predict(
        device: str,
        model: nn.Module,
        pretrained_path: str,
        batch_size: int,
        test_datasets: Dataset,
        refer_labels: list,
        criterion_name: str,
        save_option: bool
):

    test_loss = 0
    test_acc = 0
    test_precision = 0
    test_recall = 0
    test_f1 = 0

    # 加载权重
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, torch.device(device)))
        print('Successfully load pretrained model from {}'.format(pretrained_path))
    else:
        print('model parameters files is not exist!')
        sys.exit(0)
    model.to(device)

    test_dataloader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False)

    if criterion_name == 'BCELoss' or criterion_name == 'BinaryCrossEntropyLoss':
        criterion = nn.BCELoss()
    elif criterion_name == 'CELoss' or criterion_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()

    # 记录所有output，用于计算ROC和AUC
    # 至于len(refer_labels)和n_cls分开初始化是因为即使是二分类模型输出也是一个向量，而二分类下的label只是一个数。。。
    test_outputs = torch.zeros(len(test_dataloader.dataset), len(refer_labels))
    if len(refer_labels) == 2 or len(refer_labels) == 1:
        test_labels = torch.zeros(len(test_dataloader.dataset),)
    else:
        test_labels = torch.zeros(len(test_dataloader.dataset), len(refer_labels))

    model.eval()
    with tqdm(total=len(test_dataloader)) as pbar:
        pbar.set_description('loading: ')

        with torch.no_grad():
            for i, (face_img, tongue_img, label) in enumerate(test_dataloader):
                face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.long)

                output = model(tongue_img).float()
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1)

                loss = criterion(output, label)
                acc = Accuracy(output, label)
                precision, recall, f1 = Confusion_matrix(output, label, refer_labels)

                # 记录指标
                test_loss += loss.item()
                test_acc += acc
                test_precision += precision
                test_recall += recall
                test_f1 += f1
                test_outputs[i * batch_size:i * batch_size + batch_size] = output.squeeze()
                test_labels[i * batch_size:i * batch_size + batch_size] = label.squeeze()

                pbar.set_postfix({criterion_name: loss.item(), 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1})

                if save_option:
                    with open(save_predict_path, 'a', encoding='utf-8') as f:
                        for bs in range(pred.size(0)):
                            f.write('{} pred: {} label: {}\n'.format(str(i * batch_size + bs), str(pred[bs].tolist()), str(label[bs].tolist())))

                pbar.update(1)

    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)
    test_precision = test_precision / len(test_dataloader)
    test_recall = test_recall / len(test_dataloader)
    test_f1 = test_f1 / len(test_dataloader)
    if save_option:
        with open(save_predict_path, 'a', encoding='utf-8') as f:
            f.write('{}: {}  acc: {:.3f}  precision: {:.3f}  recall: {:.3f}  f1: {:.3f}'.format(criterion_name, test_loss,
                                                                                             test_acc, test_precision,
                                                                                             test_recall, test_f1))

    # 绘制ROC曲线
    fpr, tpr, roc_auc = ROC_and_AUC(test_outputs, test_labels, refer_labels, smooth=True)
    plot_ROC(fpr, tpr, roc_auc, len(refer_labels))

    return {
        'num': len(test_dataloader), 'loss': test_loss, 'acc': test_acc, 'precision': test_precision,
        'recall': test_recall, 'f1': f1
    }


if __name__ == '__main__':
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

    # 读取模型结构
    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    model = Resnet(cfg['resnet50'], 3, BasicBlock, 2)

    criterion = 'CELoss'
    print('model:\n', model)
    print('loss:', criterion)

    effect = predict(
        device=device,
        model=model,
        batch_size=batch_size,
        test_datasets=test_datasets,
        refer_labels=refer_labels,
        criterion_name=criterion,
        pretrained_path=pretrained_path,
        save_option=True
    )

    # Visualization(effect, False)
