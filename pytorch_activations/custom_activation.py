import torch
import torch.nn as nn


# https://arxiv.org/pdf/1710.05941.pdf
# https://discuss.pytorch.org/t/implementation-of-swish-a-self-gated-activation-function/8813/8
def swish(x):
    return x * torch.sigmoid(x)
