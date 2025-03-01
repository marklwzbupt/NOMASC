import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
from scipy import signal


class MNIST(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):#2598314
        length = self.data.shape[0]
        return length

    def __getitem__(self, index):
        image = self.data[index, :]
        image = (image-np.min(image))/(np.max(image) - np.min(image))
        label = self.label[index]

        return image, label


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels



