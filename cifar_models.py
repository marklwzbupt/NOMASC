import torch.nn as tnn
import torch.nn.functional as F
import torch
import numpy as np
from utils_quantization import QuantizedLinear, QuantizedLinear_cons, AveragedRangeTracker, AsymmetricQuantizer,SymmetricQuantizer

# 用encoder分别编码两个图像并且进行supercoding
class Cifar_Encoder(tnn.Module):
    def __init__(self, c):
        super(Cifar_Encoder, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer11 = tnn.Sequential(
            tnn.Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(32),
            tnn.LeakyReLU()
        )
        self.layer12 = tnn.Sequential(
            tnn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            tnn.InstanceNorm2d(64),
            tnn.LeakyReLU()
        )
        self.layer13 = tnn.Sequential(
            tnn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            tnn.InstanceNorm2d(128),
            tnn.LeakyReLU()
        )
        self.layer14 = tnn.Sequential(
            tnn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            tnn.InstanceNorm2d(256),
            tnn.LeakyReLU()
        )
        self.layer15 = tnn.Sequential(
            tnn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(512),
            tnn.LeakyReLU()
        )
        self.layer16 = tnn.Sequential(
            tnn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            tnn.InstanceNorm2d(1024),
            tnn.LeakyReLU()
        )
        self.layer17 = tnn.Sequential(
            tnn.Conv2d(1024, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            tnn.InstanceNorm2d(c),
            tnn.Tanh()
        )
        self.quant_constellation = AsymmetricQuantizer(bits = 3,
                                                       range_tracker = AveragedRangeTracker(q_level='L'))

    def forward(self, x1):
        # semantic coding of usr1 data and usr2 data

        out1 = self.layer11(x1)
        out1 = self.layer12(out1)
        out1 = self.layer13(out1)
        out1 = self.layer14(out1)
        out1 = self.layer15(out1)
        out1 = self.layer16(out1)
        out1 = self.layer17(out1)


        # normalization
        # output1 = out1 / torch.sqrt(torch.mean(torch.square(out1)))
        # output2 = out2 / torch.sqrt(torch.mean(torch.square(out2)))


        return out1

class BasicBlock(tnn.Module):
    def __init__(self, inplanes, planes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = tnn.Conv2d(inplanes, planes, stride)
        self.bn1 = tnn.InstanceNorm2d(planes)
        self.relu = tnn.ReLU(inplace=True)
        self.conv2 = tnn.Conv2d(inplanes, planes, stride)
        self.bn2 = tnn.InstanceNorm2d(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
class Cifar_Decoder(tnn.Module):
    def __init__(self,c):
        super(Cifar_Decoder, self).__init__()

        self.upsample_1 = tnn.Sequential(
            tnn.ConvTranspose2d(c, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(1024),
            tnn.LeakyReLU()
        )
        self.resblock1 = BasicBlock(1024, 1024)
        self.resblock2 = BasicBlock(1024, 1024)
        self.resblock3 = BasicBlock(1024, 1024)
        self.resblock4 = BasicBlock(1024, 1024)
        self.resblock5 = BasicBlock(1024, 1024)
        self.resblock6 = BasicBlock(1024, 1024)

        self.upsample_2 = tnn.Sequential(
            tnn.ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            tnn.InstanceNorm2d(512),
            tnn.LeakyReLU()
        )

        self.upsample_3 = tnn.Sequential(
            tnn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            tnn.InstanceNorm2d(256),
            tnn.LeakyReLU()
        )
        self.upsample_4 = tnn.Sequential(
            tnn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            tnn.InstanceNorm2d(128),
            tnn.LeakyReLU()
        )
        self.upsample_5 = tnn.Sequential(
            tnn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)),
            tnn.InstanceNorm2d(64),
            tnn.LeakyReLU()
        )
        self.upsample_6 = tnn.Sequential(
            tnn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            tnn.InstanceNorm2d(32),
            tnn.LeakyReLU()
        )
        self.upsample_7 = tnn.Sequential(
            tnn.ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            tnn.Sigmoid()
        )

    def forward(self, x):
        out = self.upsample_1(x)
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.resblock5(out)
        out = self.resblock6(out)

        out = self.upsample_2(out)
        out = self.upsample_3(out)
        out = self.upsample_4(out)
        out = self.upsample_5(out)
        out = self.upsample_6(out)
        out = self.upsample_7(out)

        return out
class Cifar_Multi_Discriminator(tnn.Module):
    def __init__(self):
        super(Cifar_Multi_Discriminator, self).__init__()
        self.D1 = tnn.Sequential(
            tnn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1)),
            tnn.LeakyReLU(),
            tnn.BatchNorm2d(64),
            tnn.ReLU(),
            tnn.Conv2d(64,1, kernel_size=(3,3), stride=(1,1)),
            tnn.Sigmoid(),
        )
        self.D2 = tnn.Sequential(
            tnn.AvgPool2d(kernel_size=(3,3), stride=(2,2)),
            tnn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1)),
            tnn.LeakyReLU(),
            tnn.BatchNorm2d(64),
            tnn.ReLU(),
            tnn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1)),
            tnn.Sigmoid(),
        )
        self.D3 = tnn.Sequential(
            tnn.AvgPool2d(kernel_size=(3,3), stride = (2,2)),
            tnn.AvgPool2d(kernel_size=(3,3), stride = (2,2)),
            tnn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1)),
            tnn.LeakyReLU(),
            tnn.BatchNorm2d(64),
            tnn.ReLU(),
            tnn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1)),
            tnn.Sigmoid(),
        )
    def forward(self,x):
        y1 = self.D1(x)
        y2 = self.D2(x)
        y3 = self.D3(x)

        y1 = y1.reshape(y1.shape[0], -1)
        y2 = y2.reshape(y2.shape[0], -1)
        y3 = y3.reshape(y3.shape[0], -1)

        y = (torch.mean(y1,1) + torch.mean(y2,1)+torch.mean(y3,1)) /3
        y = y.reshape(-1,1)
        return y


