import os
import sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   #os.path.dirname通俗的讲是返回上一级文件夹绝对路径的意思，多套一层就多返回一层
sys.path.append(path)    #将路径添加到python的搜索路径中
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from minst_loader import load_mnist, MNIST
from torch.utils.data import DataLoader
from utils import cal_grad_penalty, quantizer, gumbel_softmax_sampling, awgn_channel
import matplotlib.pyplot as plt
import cv2
#from pyldpc import make_ldpc, encode, decode, get_message
# import commpy as cpy
# import commpy.modulation as mod
from minst_models import Encoder, Decoder, Multi_Discriminator
from cifar_models import Cifar_Encoder, Cifar_Decoder, Cifar_Multi_Discriminator
from SSIM import SSIM
from modulator_models import Modulator, DeModulator
import datetime
import os
import time
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
# from utils_lsci.datasets import Vimeo, Openimage, Cityscapes_data, PALM_data
# parameters setting, user1: near user, user2: remote user
from pytorch_msssim import ms_ssim
from dataset import *
from pathlib import Path
current_path = Path(__file__).resolve().parents[0]
if str(current_path) not in sys.path:
    sys.path.append(str(current_path))
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 功率分配系数
n_rou = 0.3
f_rou = 1 - n_rou
#训练信噪比
n_snr = 15
f_snr = 5
c = 4 #输入3*256*256，得到4*64*64 下采样12倍 ;输入3*512*512，得到16*64*64 下采样12倍 
batch_size = 2
n_epoch = 20
ssim_fn = SSIM()
lambda_gp = 0.1
# model initialization
n_encoder = Cifar_Encoder(c).to(device)
f_encoder = Cifar_Encoder(c).to(device)
n_decoder = Cifar_Decoder(c).to(device)
f_decoder = Cifar_Decoder(c).to(device)
# n_Multi_dis = Cifar_Multi_Discriminator().to(device)
# f_Multi_dis = Cifar_Multi_Discriminator().to(device)
n_mod = Modulator(2).to(device)
f_mod = Modulator(2).to(device)
n_demod = DeModulator(2, 2).to(device)
f_demod = DeModulator(1, 2).to(device)

# 训练保存文件夹以及日志
def load_all_models(time, num_epoch):
    n_encoder_save = "/home/bupt-2/sms_codes/sms_codes/NOMASC/models/Open_Open/checkpoint_" +str(time) + "/n_encoder_" + str(c) + "_"+str(num_epoch)+"_.pkl"
    f_encoder_save = "/home/bupt-2/sms_codes/sms_codes/NOMASC/models/Open_Open/checkpoint_" +str(time) + "/f_encoder_" + str(c) + "_"+str(num_epoch)+"_.pkl"
    n_decoder_save = "/home/bupt-2/sms_codes/sms_codes/NOMASC/models/Open_Open/checkpoint_" +str(time) + "/n_decoder_" + str(c) + "_"+str(num_epoch)+"_.pkl"
    f_decoder_save = "/home/bupt-2/sms_codes/sms_codes/NOMASC/models/Open_Open/checkpoint_" +str(time) + "/f_decoder_" + str(c) + "_"+str(num_epoch)+"_.pkl"

    n_encoder.load_state_dict(torch.load(n_encoder_save))
    f_encoder.load_state_dict(torch.load(f_encoder_save))
    n_decoder.load_state_dict(torch.load(n_decoder_save))
    f_decoder.load_state_dict(torch.load(f_decoder_save))

'''# loading dataset minst
#images, labels = load_mnist("/home/yqk/drl_for_vne/dataset/mnist_dataset/", kind='train')
images, labels = load_mnist(r"E:\dataset\mnist_dataset", kind='train')
# images, labels = load_mnist("/home/xvxiaodong/lwz/dataset/mnist_dataset/", kind='train')
data = MNIST(images, labels)
minstloader = DataLoader(data, batch_size=batch_size, shuffle=True)

# load dataset cifar
transform = transforms.Compose(
            [transforms.ToTensor()])
cifar_dataset = torchvision.datasets.CIFAR10(root = 'E:\dataset', train = True, download = False, transform=transform)

# cifar_dataset = torchvision.datasets.CIFAR10(root = '/home/xvxiaodong/lwz/', train = True, download = False, transform=transform)
cifarloader = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
'''
# "-----------数据集加载和处理--------------"
new_size = 512
train_number = 100000
test_number = 1500
data_path = Path("/usr/Datasets/Openimage_test/testimg")
# data_path = Path("/usr/Datasets/leftImg8bit_trainvaltest_1/leftImg8bit/all_train_pics")
# data_path = Path("/usr/Datasets/PALM/PALM-Training400")

opendatset = Openimage(data_path, new_size)
# opendatset = Cityscapes_data(data_path, new_size_w)
# opendatset = PALM_data(data_path, new_size)
print('total_num_pic:', len(opendatset))
dataloader = torch.utils.data.Subset(opendatset, list(range(test_number))) 
print('dataloader_length:', len(dataloader))
test_loader = DataLoader(dataloader, batch_size= batch_size, shuffle=True, num_workers=1)

# optimizer setting
def PSNR(x, y, max_value):
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10(max_value**2 / mse)
    return torch.mean(psnr)

# mse损失函数
mse_fn = nn.MSELoss()
load_all_models("20250119-121238", 10)
#加载pre-trained 调制解调器
n_mod.load_state_dict(torch.load("/home/bupt-2/sms_codes/sms_codes/NOMASC/models/pretrained_mod/n_mod_sep.pkl"))
n_demod.load_state_dict(torch.load("/home/bupt-2/sms_codes/sms_codes/NOMASC/models/pretrained_mod/n_demod_sep.pkl"))
f_mod.load_state_dict(torch.load("/home/bupt-2/sms_codes/sms_codes/NOMASC/models/pretrained_mod/f_mod_sep.pkl"))
f_demod.load_state_dict(torch.load("/home/bupt-2/sms_codes/sms_codes/NOMASC/models/pretrained_mod/f_demod_sep.pkl"))

loop = tqdm((test_loader), total = len(test_loader))
# snr_list = [-7, -4, -1, 2, 5, 8, 11, 14, 17, 20]
snr_list = [-20, -18, -15, -13, -10, -8, -5, -2, 0, 3, 5, 8, 10, 13, 15, 18, 20, 25, 27, 30]
#snr_list = [100, 200, 300, 400, 500]
ssim1_list = []
psnr1_list = []
ssim2_list = []
psnr2_list = []
with torch.no_grad():
    for snr in snr_list:
        cnt = 0
        ssim1_all = 0
        ssim2_all = 0
        psnr1_all = 0
        psnr2_all = 0
        # for index, data_batch in enumerate(train_loader):
        loop.set_description(f'Testing: SNR [{snr}]')  
        for input in loop:
            n_snr = snr + 8
            f_snr = snr
            # 一个batch的图像,并将图像转为float类型tensor，并加载到cuda
            images_ini = input.type(torch.FloatTensor).to(device) #取出batch_size张图，最后一组不足batch_size张时，是几张就算几张。
            image1 = images_ini[0: batch_size //2, :]
            image2 = images_ini[batch_size //2 : batch_size, :]
            # image1 = image1.reshape(image1.shape[0], 3, 32, 32)
            # image2 = image2.reshape(image2.shape[0], 3, 32, 32)
            # image1 = image1.unsqueeze(0) #增加batch_size 这一维度
            # image2 = image2.unsqueeze(0) #增加batch_size 这一维度
            n_image = image2
            f_image = image1
            # 使用语义编码器对图像进行编码
            n_semantic_coding = n_encoder(n_image) * 5 + 1
            f_semantic_coding = f_encoder(f_image) * 5 + 1
            # print("n_semantic_coding.shape:", n_semantic_coding.shape)
            # print("f_semantic_coding.shape:", f_semantic_coding.shape)
            # 量化并归一化
            n_semantic_coding_q = n_encoder.quant_constellation(n_semantic_coding) / 6.6667
            f_semantic_coding_q = f_encoder.quant_constellation(f_semantic_coding) / 6.6667
            # print(torch.unique(n_semantic_coding_q,sorted=False))
            n_bits = n_semantic_coding_q.reshape(-1, 1)
            f_bits = f_semantic_coding_q.reshape(-1, 1)
            # 调制
            n_symbols = n_mod(n_bits)
            f_symbols = f_mod(f_bits)
            # 符号能量归一化
            n_symbols = n_symbols / torch.sqrt(torch.mean(torch.square(n_symbols)))
            f_symbols = f_symbols / torch.sqrt(torch.mean(torch.square(f_symbols)))
            # superimposed coding
            superimposed = np.sqrt(n_rou) * n_symbols + np.sqrt(f_rou) * f_symbols
            # 经过信道, 用户1是near user，分配较少的能量，信道条件好，用户2是far user，分配较多的能量，信道调教差
            y_n = awgn_channel(superimposed, n_snr)
            y_f = awgn_channel(superimposed, f_snr)
            # far user 直接解码
            f_bits_esti = f_demod(y_f)
            # near user
            n_bits_esti = n_demod(y_n)[:,0].unsqueeze(-1)
            # reshape 
            n_bits_esti1 = n_bits_esti.reshape(n_semantic_coding.size())
            f_bits_esti1 = f_bits_esti.reshape(f_semantic_coding.size())

            n_recon_image = n_decoder(n_bits_esti1)
            f_recon_image = f_decoder(f_bits_esti1)


            n_psnr = PSNR(n_image, n_recon_image, max_value=1)
            f_psnr = PSNR(f_image, f_recon_image, max_value=1)
            cnt += 1
            psnr1_all += n_psnr.item()
            psnr2_all += f_psnr.item()
            # ssim损失
            ssim1_all += ms_ssim(n_image, n_recon_image, data_range = 1).item()
            ssim2_all += ms_ssim(f_image, f_recon_image, data_range = 1).item()
            loop.set_postfix(ssim_n = format(ssim1_all/cnt, '.2f'), psnr_n = psnr1_all/cnt, ssim_f = format(ssim2_all/cnt, '.2f'), 
                            psnr_f = psnr2_all/cnt)

        ssim1_list.append(ssim1_all/cnt)
        ssim2_list.append(ssim2_all/cnt)
        psnr1_list.append(psnr1_all/cnt)
        psnr2_list.append(psnr2_all/cnt)
        plt.figure()
                #plt.title("snr:%f"%(snr))
        plt.subplot(221)
        plt.title("Usr1 original")
        plt.imshow(np.transpose(n_image[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
        plt.subplot(222)
        plt.title("Usr1 reconstruct")
        plt.imshow(np.transpose(n_recon_image[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
        plt.subplot(223)
        plt.title("Usr2 original")
        plt.imshow(np.transpose(f_image[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
        plt.subplot(224)
        plt.title("Usr2 reconstruct")
        plt.imshow(np.transpose(f_recon_image[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
        plt.suptitle("SNR:%f"%(snr))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0.5)
        plt.savefig("/home/bupt-2/sms_codes/sms_codes/NOMASC/save_images/open_open"+"/"+"recon_snr"+str(snr)+".jpg")
print("ssim1_list", ssim1_list)
print("===============================================================================================")
print("ssim2_list", ssim2_list)
print("===============================================================================================")
print("psnr1_list", psnr1_list)
print("===============================================================================================")
print("psnr2_list", psnr2_list)
print("===============================================================================================")
