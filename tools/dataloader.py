import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np


class MyDatasets(Dataset):
    def __init__(self, data_path: str, label: list, img_info: list, transformers: list):
        super(MyDatasets, self).__init__()
        self.data_path = data_path
        self.img_info = img_info
        self.labels = label_encoder(label)
        self.transformers = transforms.Compose(transformers)

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        # 除去换行符并以空格分割出图片名字和标签
        img_info = self.img_info[index].strip().split(' ')
        img_name = img_info[0]
        label = img_info[1]
        face_img_path = os.path.join(os.path.join(self.data_path, label), os.path.join('face', img_name))
        tongue_img_path = os.path.join(os.path.join(self.data_path, label), os.path.join('tongue', img_name))
        face_img = Image.open(face_img_path).convert('RGB')
        tongue_img = Image.open(tongue_img_path).convert('RGB')
        # 数据增强
        face_img = self.transformers(face_img)
        tongue_img = self.transformers(tongue_img)
        label = self.labels[label]

        return face_img, tongue_img, label


# 独热编码
def one_hot_encoder(label: list):
    labels = {}
    cls_num = len(label)
    for i, cls in enumerate(label):
        k = torch.zeros(cls_num)
        k[i] = 1
        labels[cls] = k

    return labels


# 数字编码，因为torch.nn.CrossEntropyLoss()会自动为其进行独热编码。。。
def label_encoder(label: list):
    labels = {}
    for i, cls in enumerate(label):
        labels[cls] = i

    return labels


def shuffle(data: list, times: int = 2):
    for _ in range(times):
        np.random.shuffle(data)
    return data



if __name__ == '__main__':
    data_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\data'
    img_names_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\data\img_names.txt'

    label = os.listdir(data_path)
    if 'img_names.txt' in label:
        label.remove('img_names.txt')

    if not os.path.exists(img_names_path):
        with open(img_names_path, 'w', encoding='utf-8') as f:
            for cls in label:
                cls_path = os.path.join(data_path, cls)
                for img in os.listdir(os.path.join(cls_path, 'tongue')):
                    f.write(img + ' ' + cls)
                    f.write('\n')
        print('Successfully generated img names file in {}!'.format(img_names_path))

    with open(img_names_path, 'r', encoding='utf-8') as f:
        img_info = f.readlines()
    print("Successfully read img names in {}!".format(data_path))

    img_info = shuffle(img_info)
    # 保存打乱顺序后的数据
    img_info = shuffle(img_info, 4)
    with open(img_names_path, 'w', encoding='utf-8') as f:
        for img in img_info:
            f.write(img)
    print('Successfully shuffle img names file in {}!'.format(img_names_path))

    transformers = [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
    test_datasets = MyDatasets(data_path, label, img_info, transformers)
    face_img, tongue_img, label = test_datasets.__getitem__(1)
    plt.subplot(1, 2, 1)
    plt.imshow(face_img.permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(tongue_img.permute(1, 2, 0))
    plt.show()
    print(label)
