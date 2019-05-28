
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torchvision.transforms import functional as F
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt

def imshow(images, title=None):
    images = images.numpy().transpose((1, 2, 0))  # (h, w, c)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)
    plt.imshow(images)