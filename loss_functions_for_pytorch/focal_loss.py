import torch
import torch.nn as nn
import torch.nn.functional as F

# paper https://arxiv.org/pdf/1708.02002.pdf
# 解説 https://qiita.com/agatan/items/53fe8d21f2147b0ac982
# https://datascience.stackexchange.com/questions/31685/weighted-cross-entropy-for-imbalanced-dataset-multiclass-classification

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()

