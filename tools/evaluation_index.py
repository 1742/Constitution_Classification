import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from scipy import interp
import json


def Accuracy(predict: torch.Tensor, label: torch.Tensor, refer_label: list):
    if len(refer_label) == 2:
        predict = (predict > 0.5).long().cpu()
    else:
        # 标签用的序号编码，不是独热
        predict = torch.argmax(torch.softmax(predict, dim=1), dim=1).cpu()
    label = label.long().cpu()

    acc = torch.sum(predict == label).long()

    return float(acc) / predict.size(0)


def Confusion_matrix(predict: torch.Tensor, label: torch.Tensor, refer_labels: list):
    # 防止出现分母为0的情况
    smooth = 1e-5

    if len(refer_labels) == 2:
        predict = (predict > 0.5).long().cpu()
    else:
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


def ROC_and_AUC(predict: torch.Tensor, label: torch.Tensor, refer_label: list = None, smooth: bool = False, num_points: int = 50):
    """
    计算该数据集的TRP、FRP、AUC

    :param predict:
        未经softmax或sigmoid的模型输出
    :param label:
        实际标签，序号编码，内部会自动转为独热
    :param refer_label:
        参考标签
    :param smooth:
        是否平滑ROC曲线，使用线性插值平滑
    :param num_points:
        选择平滑ROC后生效，插值点数量

    :return:
        各类别各阈值下的TRP、FRP，以及各类别AUC
    """

    n_cls = len(refer_label)
    # 因为label_binarize不会将二分类标签做成二维向量，之后使用roc_curve时会报错。。。
    if n_cls == 2:
        n_cls = 1

    if len(refer_label) == 2:
        predict = (predict > 0.5).long().cpu()
    else:
        predict = torch.softmax(predict, dim=1).cpu()
    predict = np.array(predict)
    label = label.long().cpu()

    if refer_label:
        label = label_binarize(label, classes=refer_label)
    else:
        label = label_binarize(label, classes=torch.max(label))

    # 分别计算各类别的ROC曲线，若为二分类，只有一条曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_cls):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], predict[:, i])
        if smooth:
            tpr[i] = np.interp(np.linspace(0, 1, num_points), fpr[i], tpr[i])
            tpr[i][0] = 0.0
            fpr[i] = np.linspace(0, 1, num_points)
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def Visualization(evaluation, train: bool):
    index = list(evaluation.keys())
    if train:
        index.remove('epoch')

    epoch = range(1, evaluation['epoch']+1)
    for i, k in enumerate(index):
        plt.plot(epoch, evaluation[k][0], label='train')
        plt.plot(epoch, evaluation[k][1], label='val')
        plt.title('train' + k)
        plt.xlabel('epoch')
        plt.ylabel(k)
        plt.legend()
        plt.show()


def plot_ROC(fpr: [dict, list], tpr: [dict, list], roc_auc: [dict, list], n_cls: int, model_num: int = 1, model_name: list = None):
    if n_cls == 2:
        n_cls = 1

    # 创建画布
    plt.figure()
    # 设定线宽
    lw = 2
    if model_num != 1:
        for k in range(model_num):
            if model_name:
                name = model_name[k]
            else:
                name = k
            for i in range(n_cls):
                plt.plot(fpr[k][i], tpr[k][i], lw=lw, label='ROC curve of class {}, model: {}  (AUC = {:.2f})'.format(i + 1, name, roc_auc[k][i]))
    else:
        for i in range(n_cls):
            plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class %d (AUC = %0.2f)' % (i + 1, roc_auc[i]))
    # 参考线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # 坐标大小
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # 横纵坐标和标题，标签
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    predict = torch.rand((16, 1))
    label = torch.randint(2, (16, 1))
    refer_labels = [0, 1]

    # 测试代码
    acc = Accuracy(predict, label, refer_labels)
    print(acc)
    P, R, F1 = Confusion_matrix(predict, label, refer_labels)
    print(P, R, F1)
    fpr, tpr, roc_auc = ROC_and_AUC(predict, label, refer_labels, smooth=True, num_points=50)
    plot_ROC(fpr, tpr, roc_auc, len(refer_labels))

    # effect_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\runs\pretreatment_resnet34\train\effect.json'
    # save_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\runs\pretreatment_resnet34\train'
    #
    # with open(effect_path, 'r', encoding='utf-8') as f:
    #     effect = json.load(f)
    #
    # Visualization(effect, True)


