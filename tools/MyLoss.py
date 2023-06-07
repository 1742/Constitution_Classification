import torch
from torch import nn


class M_and_D_UncLoss(nn.Module):
    def __init__(self, num_classess: int, device: str):
        super(M_and_D_UncLoss, self).__init__()
        self.device = device
        if num_classess == 2:
            self.cls_loss = nn.BCELoss()
        else:
            self.cls_loss = nn.CrossEntropyLoss()

        self.unc_weight_cls_1, self.unc_weight_cls_2 = self.init_unc_weight()
        self.unc_weight_var_1, self.unc_weight_var_2 = self.init_unc_weight()

    def forward(self, pred, labels, var):
        cls_loss = self.cls_loss(pred, labels)
        var = var.mean()

        # 模型不确定性加权损失
        unc_cls_loss = self.unc_weight_cls_1 * cls_loss + self.unc_weight_cls_2
        # 数据不确定性加权损失
        unc_var_loss = self.unc_weight_var_1 * var + self.unc_weight_var_2
        # 模型不确定性、数据不确定性加权损失
        total_loss = unc_cls_loss + unc_var_loss

        return total_loss.sum()

    def init_unc_weight(self):
        weights = torch.rand(1, device=self.device)
        unc_weight_1 = 0.5 * torch.pow(weights, -2)
        unc_weight_2 = torch.log(1 + torch.pow(weights, 2))
        # 确保初始损失不会太大。。。
        if unc_weight_1 > 10:
            unc_weight_1 /= 10.0
        if unc_weight_2 > 10:
            unc_weight_2 /= 10.0

        unc_weight_1 = nn.Parameter(unc_weight_1)
        unc_weight_2 = nn.Parameter(unc_weight_2)

        return unc_weight_1, unc_weight_2


if __name__ == '__main__':
    pred = torch.rand((64, 1))
    labels = torch.randint(2, (64, 1)).float()
    var = torch.rand((64, 1))

    criterion = M_and_D_UncLoss(2, 'cpu')

    loss = criterion(pred, labels, var)
    optim = torch.optim.Adam(criterion.parameters(), lr=1e-4, weight_decay=1e-8)
    print(loss)
    loss.backward()
    optim.step()

