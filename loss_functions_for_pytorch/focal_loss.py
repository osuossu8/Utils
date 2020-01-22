import torch
import torch.nn as nn
import torch.nn.functional as F

# paper https://arxiv.org/pdf/1708.02002.pdf
# 解説 https://qiita.com/agatan/items/53fe8d21f2147b0ac982
# https://datascience.stackexchange.com/questions/31685/weighted-cross-entropy-for-imbalanced-dataset-multiclass-classification
# from https://www.kaggle.com/hmendonca/kaggle-pytorch-utility-script


class FocalLoss(nn.Module):
    ''' cross entropy focal loss '''
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
            self.alpha = self.alpha.type(inputs.type(), non_blocking=True) # fix type and device
            alpha = self.alpha[targets]
        else:
            alpha = 1.

        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = alpha * (1-pt)**self.gamma * CE_loss

        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()
        return F_loss

