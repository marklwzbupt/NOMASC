import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

class Modulator(torch.nn.Module):
    def __init__(self, dim):
        super(Modulator, self).__init__()
        self.l1 = torch.nn.Linear(1, dim)

    def forward(self, x):
        out = self.l1(x)
        return out

class DeModulator(torch.nn.Module):
    def __init__(self, k, dim):
        super(DeModulator, self).__init__()
        self.l1 = torch.nn.Linear(dim, 128)
        self.l2 = torch.nn.Linear(128, 64)
        self.l3 = torch.nn.Linear(64, 32)
        self.l4 = torch.nn.Linear(32, k)


    def forward(self, x):
        out1 = self.l1(x)
        out2 = F.tanh(out1)
        out3 = self.l2(out2)
        out4 = F.tanh(out3)
        out5 = self.l3(out4)
        out6 = F.tanh(out5)
        out7 = self.l4(out6)
        #out1 = F.tanh(out1)

        return out7
