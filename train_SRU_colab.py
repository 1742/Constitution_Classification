import sys

import torch
from torch import nn
from torch.functional import F
from model.resnet.resnet import Resnet, pretrained_Resnet
from model.SRU.SRU import SRU
from tools.MyLoss import Any_Uncertainty_Loss

from torch.utils.data import Dataset, DataLoader
from tools.dataloader import MyDatasets, shuffle, label_encoder
from torchvision import transforms
from tools.Mytransforms import Resize, RandomHorizontalFlip, RandomRotation, ToTensor, Compose
import numpy as np

import os
import json
from tqdm import tqdm
from tools.evaluation_index import Accuracy, Confusion_matrix, Visualization
import sys


data_path = r'content/drive/MyDrive/Colab Notebooks/Constitution_Classification/data'
data_path_txt = r'content/drive/MyDrive/Colab Notebooks/Constitution_Classification/data/img_names.txt'
cfg_file = r'content/drive/MyDrive/Colab Notebooks/Constitution_Classification/model/config.json'
pretrained_path = r'content/drive/MyDrive/Colab Notebooks/Constitution_Classification/checkpoints/resnet34-b627a593.pth'
save_path = r'content/drive/MyDrive/Colab Notebooks/Constitution_Classification/model/SRU/SRU.pth'
effect_path = r'content/drive/MyDrive/Colab Notebooks/Constitution_Classification/runs/SRU/train'


learning_rate = 1e-3
weight_decay = 1e-8
epochs = 50
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('The train will run in {} ...'.format(device))
save_option = True


def train(
        device: str,
        model: nn.Module,
        trian_datasets: Dataset,
        val_datasets: Dataset,
        refer_labels: list,
        batch_size: int,
        epochs: int,
        lr: float,
        weight_decay: float,
        optim: str,
        criterion_name: str,
        save_option: bool,
        lr_schedule: dict = None,
        pretrained_path: str = None,
):

    # 返回指标
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []

    # 加载权重
    if pretrained_path:
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, torch.device(device)))
            print('Successfully load pretrained model from {}'.format(pretrained_path))
        else:
            print('model parameters files is not exist!')
            sys.exit(0)
    model.to(device)

    train_dataloader = DataLoader(trian_datasets, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True)

    if optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 各个损失函数需要的标签形式不同，要去dataloader里手动切换......
    if criterion_name == 'Any_Uncertainty_Loss':
        criterion = Any_Uncertainty_Loss(len(refer_labels))

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    elif lr_schedule['name'] == 'StepLR':
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_schedule['step_size'], gamma=lr_schedule['gamma'])
    elif lr_schedule['name'] == 'ExponentialLR':
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_schedule['gamma'])

    for epoch in range(epochs):
        # 训练
        model.train()

        per_train_loss = 0
        per_train_acc = 0
        per_train_precision = 0
        per_train_recall = 0
        per_train_f1 = 0
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (face_img, tongue_img, label) in enumerate(train_dataloader):
                # face_img = face_img.to(device, dtype=torch.float)
                tongue_img = tongue_img.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)

                output, x_, mu, sigma = model(tongue_img)

                loss = criterion(x_, mu, sigma, label)

                acc = Accuracy(output, label, refer_labels)
                precision, recall, f1 = Confusion_matrix(output, label, refer_labels)

                # 记录每批次平均指标
                per_train_loss += loss.item()
                per_train_acc += acc
                per_train_precision += precision
                per_train_recall += recall
                per_train_f1 += f1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_schedule.step()

                pbar.set_postfix({criterion_name: loss.item(), 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1})
                pbar.update(1)

        # 记录每训练次平均指标
        train_loss.append(per_train_loss / len(train_dataloader))
        train_acc.append(per_train_acc / len(train_dataloader))
        train_precision.append(per_train_precision / len(train_dataloader))
        train_recall.append(per_train_recall / len(train_dataloader))
        train_f1.append(per_train_f1 / len(train_dataloader))

        # 验证
        model.eval()

        per_val_loss = 0
        per_val_acc = 0
        per_val_precision = 0
        per_val_recall = 0
        per_val_f1 = 0
        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (face_img, tongue_img, label) in enumerate(val_dataloader):
                    # face_img = face_img.to(device, dtype=torch.float)
                    tongue_img = tongue_img.to(device, dtype=torch.float)
                    label = label.to(device, dtype=torch.float)

                    output, x_, mu, sigma = model(tongue_img)

                    loss = criterion(x_, mu, sigma, label)

                    acc = Accuracy(output, label, refer_labels)
                    precision, recall, f1 = Confusion_matrix(output, label, refer_labels)
                    per_val_precision += precision
                    per_val_recall += recall
                    per_val_f1 += f1

                    # 记录指标
                    per_val_loss += loss.item()
                    per_val_acc += acc

                    pbar.set_postfix({criterion_name: loss.item(), 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1})
                    pbar.update(1)

        # 记录每训练次平均指标
        val_loss.append(per_val_loss / len(val_dataloader))
        val_acc.append(per_val_acc / len(val_dataloader))
        val_precision.append(per_val_precision / len(val_dataloader))
        val_recall.append(per_val_recall / len(val_dataloader))
        val_f1.append(per_val_f1 / len(val_dataloader))

    if save_option:
        torch.save(model.state_dict(), save_path.format(model_name))
        print('Successfully saved model weights in {}'.format(save_path))

    return {
        'epoch': epochs, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],
        'precision': [train_precision, val_precision], 'recall': [train_recall, val_recall],
        'f1': [train_f1, val_f1]
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
    print("Successfully read img names from {}!".format(data_path))

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

    train_transformers = [
        Resize((224, 224)),
        RandomHorizontalFlip(),
        RandomRotation((0, 15)),
        ToTensor()
    ]
    val_transformers = [
        Resize((224, 224)),
        ToTensor()
    ]

    train_datasets = MyDatasets(data_path, labels, train_data_info, train_transformers)
    val_datasets = MyDatasets(data_path, labels, test_data_info, val_transformers)

    # 读取模型结构
    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    model_name = 'SRU'
    model = SRU(cfg['resnet34'], 3, 2)
    pretrained_path = None

    optimizer = 'Adam'
    criterion = 'Any_Uncertainty_Loss'
    lr_schedule = {'name': 'StepLR', 'step_size': 5, 'gamma': 0.9}
    # lr_schedule = None
    print('model:\n', model)
    print('epoch:', epochs)
    print('batch size:', batch_size)
    print('learning rate:', learning_rate)
    print('weight decay:', weight_decay)
    print('loss:', criterion)
    print('optimizer:', optimizer)
    print('lr_schedule:', lr_schedule)

    effect = train(
        device=device,
        model=model,
        trian_datasets=train_datasets,
        val_datasets=val_datasets,
        refer_labels=refer_labels,
        batch_size=batch_size,
        epochs=epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        optim=optimizer,
        criterion_name=criterion,
        save_option=save_option,
        lr_schedule=lr_schedule,
        pretrained_path=pretrained_path,
    )

    if not os.path.exists(effect_path):
        os.mkdir(effect_path)
    with open(effect_path+'\\effect.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(effect))

    Visualization(effect, True)

