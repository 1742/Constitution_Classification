import os
import sys

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
import math
from PIL import Image
import cv2
import matplotlib.pyplot as plt


# 为了能同时对图片和标签进行相同处理，覆写了一部分transforms的代码
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = F.resize(image, self.size)
        if target is not None:
            target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
            return image, target
        return image


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
                return image, target
            return image
        if target:
            return image, target
        return image


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        # emmmm这个插值方法没法改，默认双线性插值

    @staticmethod
    def get_params(img: torch.Tensor, scale: list, ratio: list):
        _, height, width = F.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, image, target=None):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        image = F.resized_crop(image, i, j, h, w, self.size)
        if target:
            target = F.resized_crop(target, i, j, h, w, self.size)
            return image, target
        return image


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if target:
                target = F.vflip(target)
                return image, target
            return image
        if target:
            return image, target
        return image


class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def get_params(self, degrees: list):
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def __call__(self, image, target=None):
        angle = self.get_params(self.degrees)

        image = F.rotate(image, angle)
        if target:
            target = F.rotate(target, angle)
            return image, target
        return image


class ToTensor(object):
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            if target.mode == 'P':
                target = torch.as_tensor(np.array(target), dtype=torch.int64)
            else:
                target = F.to_tensor(target)
            return image, target
        return image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is not None:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target
        else:
            for t in self.transforms:
                image = t(image)
            return image


class RGBToHSV(object):
    def __init__(self):
        self.smooth = 1e-5

    def __call__(self, image, target=None):
        image_hsv = self.rgb2hsv(image)
        if target is not None:
            target_hsv = self.rgb2hsv(target)
            return image_hsv, target_hsv
        return image_hsv

    def rgb2hsv(self, image):
        if image.mode != 'RGB':
            return image

        image = np.array(image, dtype=np.float32) / 255
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        Cmax = np.max(image, axis=2)
        Cmin = np.min(image, axis=2)
        d = Cmax - Cmin
        # 计算H
        H_R = np.array(R == Cmax, dtype=np.float32) * ((G - B) / (d + self.smooth))
        H_R_mask = np.array(H_R < 0, dtype=np.float32) * 360
        H_R = H_R + H_R_mask
        H_G = np.array(G == Cmax, dtype=np.float32) * ((B - R) / (d + self.smooth) + 120)
        H_B = np.array(B == Cmax, dtype=np.float32) * ((R - G) / (d + self.smooth) + 240)
        H = (H_R + H_G + H_B) / 2
        # 计算S
        S_0 = np.array(Cmax == 0, dtype=np.float32)
        S_1 = np.array(Cmax != 0, dtype=np.float32) * (d / (Cmax + self.smooth))
        S = (S_0 + S_1) * 255
        # 计算V
        V = Cmax * 255

        image_hsv = np.array([H, S, V], dtype='uint8').transpose(1, 2, 0)
        image_hsv = Image.fromarray(image_hsv)

        return image_hsv


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.colorjitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target=None):
        image = self.colorjitter(image)
        if target is not None:
            if target.mode == 'RGB':
                target = self.colorjitter(target)
            return image, target
        return image


if __name__ == '__main__':
    data_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\data\sx'
    test_img_name = '20722.png'
    test_face_path = os.path.join(data_path, 'face\\' + test_img_name)
    test_tongue_path = os.path.join(data_path, 'tongue\\' + test_img_name)

    test_face_img = Image.open(test_face_path).convert('RGB')
    test_tongue_img = Image.open(test_tongue_path).convert('RGB')

    transformers = Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation((0, 15)),
        ToTensor()
    ])

    test_face_img_, test_tongue_img_ = transformers(test_face_img, test_tongue_img)

    plt.subplot(221)
    plt.imshow(test_face_img)

    plt.subplot(222)
    plt.imshow(test_face_img_.permute(1, 2, 0))

    plt.subplot(223)
    plt.imshow(test_tongue_img)

    plt.subplot(224)
    plt.imshow(test_tongue_img_.permute(1, 2, 0))

    plt.show()


