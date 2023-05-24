import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import json


def Accuracy(predict: torch.Tensor, label: torch.Tensor):
    # 标签用的序号编码，不是独热
    predict = torch.argmax(torch.softmax(predict, dim=1), dim=1).cpu()
    label = label.long().cpu()

    acc = torch.sum(predict == label).long()

    return float(acc) / predict.size(0)


def Confusion_matrix(predict: torch.Tensor, label: torch.Tensor, refer_labels: list):
    # 防止出现分母为0的情况
    smooth = 1e-17

    predict = torch.argmax(torch.softmax(predict, dim=1), dim=1).cpu()
    label = label.long().cpu()
    c_m = confusion_matrix(y_pred=predict, y_true=label, labels=refer_labels)

    Precision = 0
    Recall = 0
    # 分别计算各类别的Precision, Recall
    for cls in range(c_m.shape[0]):
        Precision += c_m[cls, cls] / (np.sum(c_m[:, cls]) + smooth)
        Recall += c_m[cls, cls] / np.sum(c_m[cls, :] + smooth)

    Precision /= c_m.shape[0]
    Recall /= c_m.shape[0]
    F1 = (2 * Precision * Recall) / (Precision + Recall + smooth)

    return Precision, Recall, F1


def Visualization(evaluation, train: bool, save_option: str = None):
    index = list(evaluation.keys())
    index.remove('epoch')
    figure_num = len(index)
    epoch = range(1, evaluation['epoch']+1)
    for i, k in enumerate(index):
        plt.subplot(1, figure_num, i+1)
        if train:
            plt.plot(epoch, evaluation[k][0], label='train')
            plt.plot(epoch, evaluation[k][1], label='val')
            plt.title('train' + k)
            plt.xlabel('epoch')
            plt.ylabel(k)
            plt.legend()
        else:
            plt.plot(epoch, evaluation[k], label='test')
            plt.title('test' + k)
            plt.xlabel('epoch')
            plt.ylabel(k)
            plt.legend()

    if save_option:
        plt.savefig(save_option)
        print('Successfully save config in {}'.format(save_option))

    plt.show()


if __name__ == '__main__':
    predict = torch.rand((16, 3))
    label = torch.randint(3, (16,))
    refer_labels = [0, 1, 2]
    acc = Accuracy(predict, label)
    print(acc)
    P, R, F1 = Confusion_matrix(predict, label, refer_labels)
    print(P, R, F1)
