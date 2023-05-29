import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys


def generate_tongue(img_path, target_path, save_path=None):
    img = np.array(Image.open(img_path).convert('RGB'))
    target = np.array(Image.open(target_path).convert('P')) != 0
    target = np.expand_dims(target, axis=-1).astype(np.uint8)

    tongue = Image.fromarray(img * target).convert('RGB')
    if save_path:
        tongue.save(save_path)
    else:
        return tongue


def generate_face(img_path, target_path, save_path=None):
    img = np.array(Image.open(img_path).convert('RGB'))
    target = np.array(Image.open(target_path).convert('P')) != 0
    target = np.expand_dims(target, axis=-1).astype(np.uint8)

    face = Image.fromarray(img * target).convert('RGB')
    if save_path:
        face.save(save_path)
    else:
        return face


if __name__ == '__main__':
    data_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\data'
    img_names_txt = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\data\img_names.txt'

    with open(img_names_txt, 'r', encoding='utf-8') as f:
        info = f.readlines()
        with tqdm(total=len(info)) as pbar:
            for img_info in info:
                img_name, label = img_info.strip().split(' ')
                img_path = os.path.join(data_path, label + '\\image\\{}'.format(img_name))
                target_path = os.path.join(data_path, label + '\\face\\{}'.format(img_name))
                # target_path = os.path.join(data_path, label + '\\tongue\\{}'.format(img_name))

                # save_path = os.path.join(data_path, label + '\\tongue_\\{}'.format(img_name))
                save_path = os.path.join(data_path, label + '\\face_\\{}'.format(img_name))
                generate_tongue(img_path, target_path, save_path)

                pbar.update(1)
