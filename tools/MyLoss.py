import sys

import torch
from torch import nn
from torch.functional import F


class CELoss(object):
    def __init__(self, n_cls: int):
        super(CELoss, self).__init__()
        self.n_cls = n_cls

    def __call__(self, preds: torch.Tensor, labels: torch.Tensor):
        # 根据输入的标签类型转化成序号标签
        if labels.max() == 1 and self.n_cls != 2:
            labels = torch.argmax(labels, dim=1).long().cpu()
        else:
            labels = labels.long().cpu()

        return F.nll_loss(preds, labels)


class Any_Uncertainty_Loss(nn.Module):
    def __init__(self, n_cls: int, n_sample: int = 10):
        super(Any_Uncertainty_Loss, self).__init__()
        self.n_cls = n_cls
        self.celoss = CELoss(n_cls)
        self.n_sample = n_sample

    def forward(self, x_: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, label: torch.Tensor):
        """
        计算嵌入了不确定性的损失

        :param x_:
            未经LogSoftmax的全连接层输出
        :param mu:
            模型输出的任意不确定性“均值”
        :param sigma:
            模型输出的任意不确定性“方差”
        :param label:
            标签，序号编码

        :return:
            任意不确定性损失
        """
        x_ = x_.cpu()
        label = label.long().cpu()
        # 在各类别上加n_sample次高斯分布
        x_gauss_distrib_sample_prob = torch.zeros((x_.size(0), self.n_sample, self.n_cls))
        for t in range(self.n_sample):
            epsilon = torch.randn(sigma.size())
            epsilon = mu + torch.mul(sigma, epsilon)
            x_gauss_distrib_sample_prob[:, t, :] = F.softmax(x_ * epsilon, dim=1)

        logits = torch.mean(x_gauss_distrib_sample_prob, 1)
        unc_loss = F.nll_loss(torch.log(logits), label)

        return unc_loss

