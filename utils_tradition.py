from torch.utils.data import DataLoader, Dataset
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn as nn
import torchvision.transforms as transforms
from SSIM import SSIM
#from ssim import ssim as SSIM
import commpy.modulation as mod
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
# QPSK调制
modem = mod.PSKModem(4)
def quan_to_bit_mappping(quantized):
    quantized = quantized * 10
    quantized = torch.round(quantized).to(dtype = torch.int64)
    # print(quantized)
    quan_list = [-5, 0, 5, 10]
    for j in reversed(range(len(quan_list))):
        quantized = torch.where(quantized == quan_list[j], j, quantized)
    quantized = np.reshape(quantized.cpu().numpy(), [-1])
    b_bits = []
    for n in range(len(quantized)):
        bs = [int(x) for x in bin(quantized[n])[2:]]
        b_bits.extend([0] * (2 - len(bs)))
        b_bits.extend(bs)
    return b_bits

def modulation(bits):
    # b_bits = []
    # for n in range(len(bits)):
    #     bs = [int(x) for x in bin(bits[n])[2:]]
    #     b_bits.extend([0] * (3 - len(bs)))
    #     b_bits.extend(bs)
    #modem = mod.QAMModem(4)
    symbol = modem.modulate(bits)
    return symbol

def demodulation(superimposed):
    bits = modem.demodulate(superimposed, 'hard')
    return bits
def bit_to_quan_mapping(bits):
    quan_list = [-0.5, 0, 0.5, 1]
    q_quan = []
    for j in range(len(bits)//2):
        dec = bits[2*j] * 2 + bits[2*j+1]
        # if dec == 7:
        #     dec = 3
        q_quan.append(quan_list[dec])
    return torch.from_numpy(np.array(q_quan)).float().cuda()

def sic_process(bit1_esti, superimposed, f_rou):
    strong_recon = modem.modulate(bit1_esti)
    residual = superimposed - np.sqrt(f_rou) * strong_recon
    return residual
def awgn_channel_np(input, snr):
    shape = np.shape(input)
    power = np.real(np.mean(input * np.conj(input)))
    std =  np.sqrt(10 ** (-snr / 10) * power)
    #print(std)
    noise = (np.random.randn(shape[0]) + 1j * np.random.randn(shape[0])) / (np.sqrt(2))
    return input + std * noise