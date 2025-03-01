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
current_time = time.strftime("%Y%m%d-%H%M%S")
# result_dir = "../models_sms/open_open/checkpoint_" + current_time
result_dir = "/home/bupt-2/sms_codes/sms_codes/NOMASC/models/Open_Open/checkpoint_" + current_time
if not os.path.exists(result_dir):
    os.makedirs(result_dir) #os.makedirs() 会尝试创建一个目录，而不是文件。
train_log_filename = "train_log.txt"
train_log_filepath = os.path.join(result_dir, train_log_filename) #创建文件，所以不能用makedirs
train_log_filepath = Path(train_log_filepath)
# 你可以使用 Path().touch() 来确保文件存在，如果文件不存在，它会创建空文件
train_log_filepath.touch(exist_ok=True)
train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [f_mse] {fmse_str} [n_mse] {nmse_str} [n_ssim] {nssim_str} [f_ssim] {fssim_str} [n_psnr] {npsnr_str} [f_psnr] {fpsnr_str}\n"

def save_all_models(num_epoch):
    n_encoder_save = result_dir + "/n_encoder_" + str(c) + "_" + str(num_epoch) + "_.pkl"
    f_encoder_save = result_dir + "/f_encoder_" + str(c) + "_" + str(num_epoch) + "_.pkl"
    n_decoder_save = result_dir + "/n_decoder_" + str(c) + "_" + str(num_epoch) + "_.pkl"
    f_decoder_save = result_dir + "/f_decoder_" + str(c) + "_" + str(num_epoch) + "_.pkl"
    # n_Multi_dis_save = result_dir + "/n_md.pkl"
    # f_Multi_dis_save = result_dir + "/f_md.pkl"
    torch.save(n_encoder.state_dict(), n_encoder_save)
    torch.save(n_decoder.state_dict(), n_decoder_save)
    torch.save(f_encoder.state_dict(), f_encoder_save)
    torch.save(f_decoder.state_dict(), f_decoder_save)

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
data_path = Path("/usr/Datasets/Openimage")
# data_path = Path("/usr/Datasets/leftImg8bit_trainvaltest_1/leftImg8bit/all_train_pics")
# data_path = Path("/usr/Datasets/PALM/PALM-Training400")

opendatset = Openimage(data_path, new_size)
# opendatset = Cityscapes_data(data_path, new_size_w)
# opendatset = PALM_data(data_path, new_size)
print('total_num_pic:', len(opendatset))
dataloader = torch.utils.data.Subset(opendatset, list(range(train_number)))  
print('dataloader_length:', len(dataloader))
train_loader = DataLoader(dataloader, batch_size= batch_size, shuffle=True, num_workers=1)

# optimizer setting
opt_tran = torch.optim.Adam(
    [{'params': n_encoder.parameters(), 'lr': 0.0001},
     {'params': f_encoder.parameters(), 'lr': 0.0001},
     {'params': n_decoder.parameters(), 'lr': 0.0001},
     {'params': f_decoder.parameters(), 'lr': 0.0001}
     ])
def PSNR(x, y, max_value):
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10(max_value**2 / mse)
    return torch.mean(psnr)
# MD1_optimizer = torch.optim.Adam(n_Multi_dis.parameters(), lr = 0.0001)
# MD2_optimizer = torch.optim.Adam(f_Multi_dis.parameters(), lr = 0.0001)
# loss 列表
e_mse1_list = []
e_mse2_list = []
e_ssim1_list = []
e_ssim2_list = []
# mse损失函数
mse_fn = nn.MSELoss()

#加载pre-trained 调制解调器
n_mod.load_state_dict(torch.load("/home/bupt-2/sms_codes/sms_codes/NOMASC/models/pretrained_mod/n_mod_sep.pkl"))
n_demod.load_state_dict(torch.load("/home/bupt-2/sms_codes/sms_codes/NOMASC/models/pretrained_mod/n_demod_sep.pkl"))
f_mod.load_state_dict(torch.load("/home/bupt-2/sms_codes/sms_codes/NOMASC/models/pretrained_mod/f_mod_sep.pkl"))
f_demod.load_state_dict(torch.load("/home/bupt-2/sms_codes/sms_codes/NOMASC/models/pretrained_mod/f_demod_sep.pkl"))

best_fssim = 0.2
best_nssim = 0.2

# 训练过程
for epoch in range(n_epoch):
    #dataloader_iterator = iter(minstloader)
    mse1_list = []
    mse2_list = []
    ssim1_list = []
    ssim2_list = []
    psnr1_list = []
    psnr2_list = []
    # for index, data_batch in enumerate(train_loader):
    loop = tqdm((train_loader), total = len(train_loader))
    loop.set_description(f'Training: Epoch [{epoch+1}/{n_epoch}]')  
    for input in loop:
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

        # L2损失
        image1_L2 = ((n_recon_image - n_image) ** 2).mean()
        image2_L2 = ((f_recon_image - f_image) ** 2).mean()
        L2 = image1_L2 + image2_L2
        mse1_list.append(image1_L2.detach().cpu().numpy())
        mse2_list.append(image2_L2.detach().cpu().numpy())
        n_psnr = PSNR(n_image, n_recon_image, max_value=1)
        f_psnr = PSNR(f_image, f_recon_image, max_value=1)
        psnr1_list.append(n_psnr.item())
        psnr2_list.append(f_psnr.item())
        # ssim损失
        image1_ssim_loss = 1 - ms_ssim(n_image, n_recon_image, data_range = 1)
        image2_ssim_loss = 1 - ms_ssim(f_image, f_recon_image, data_range = 1)
        ssim_loss = image1_ssim_loss + image2_ssim_loss
        ssim1_list.append(ms_ssim(n_image, n_recon_image, data_range = 1).item())
        ssim2_list.append(ms_ssim(f_image, f_recon_image, data_range = 1).item())
        "patch loss 图像真实性判别器 判别器需要识别重建图像和原始图像，生成器与之对抗，尽量缩小重建图像与原始图像的差别，不让判别器识别"
        loop.set_postfix(ssim_n = format(np.mean(ssim1_list), '.2f'), psnr_n = np.mean(np.array(psnr1_list)), ssim_f = format(np.mean(ssim2_list), '.2f'), 
                         psnr_f = np.mean(np.array(psnr2_list)))
        # 将原图像输入到判别器中
        # real_output1 = n_Multi_dis(n_image)
        # # 将重建图像输入到判别器中，这里detach了
        # fake_output_d1 = n_Multi_dis(n_recon_image.detach())
        # # 将重建图像输入到判别器中，这里没有detach
        # fake_output_g1 = n_Multi_dis(n_recon_image)

        # # 将原图像输入到判别器中
        # real_output2 = f_Multi_dis(f_image)
        # # 将重建图像输入到判别器中，这里detach了
        # fake_output_d2 = f_Multi_dis(f_recon_image.detach())
        # # 将重建图像输入到判别器中，这里没有detach
        # fake_output_g2 = f_Multi_dis(f_recon_image)

        # "WGAN-GP loss"
        # # generator的loss计算，也就是生成图像的平均值取负数
        # g_loss1 = -torch.mean(fake_output_g1)
        # g_loss2 = -torch.mean(fake_output_g2)
        # g_loss = g_loss1 + g_loss2
        # # 梯度惩罚项
        # grad_penalty1 = cal_grad_penalty(n_Multi_dis, n_image.data, n_recon_image.data)
        # grad_penalty2 = cal_grad_penalty(f_Multi_dis, f_image.data, f_recon_image.data)
        # # 判别器的loss
        # d_loss1 = -torch.mean(real_output1) + torch.mean(
        #     fake_output_d1) + lambda_gp * grad_penalty1  # 改进2、生成器和判别器的loss不取log
        # d_loss2 = -torch.mean(real_output2) + torch.mean(
        #     fake_output_d2) + lambda_gp * grad_penalty2

        "Update"
        # 更新编码器和生成器的参数
        # ae_loss = L2 + 0.01 * ssim_loss + 0.0001 * g_loss
        ae_loss = L2 + 0.01 * ssim_loss 
        opt_tran.zero_grad()
        ae_loss.backward()
        opt_tran.step()

        # # 更新判别器的权重
        # "train patch D"
        # md1_loss = d_loss1 * 0.1
        # MD1_optimizer.zero_grad()
        # md1_loss.backward()
        # MD1_optimizer.step()

        # md2_loss = d_loss2 * 0.1
        # MD2_optimizer.zero_grad()
        # md2_loss.backward()
        # MD2_optimizer.step()

        perce_loss = 0
       
    mse1 = np.mean(np.array(mse1_list))
    mse2 = np.mean(np.array(mse2_list))
    ssim1 = np.mean(np.array(ssim1_list))
    ssim2 = np.mean(np.array(ssim2_list))
    psnr1 = np.mean(np.array(psnr1_list))
    psnr2 = np.mean(np.array(psnr2_list))
    e_mse1_list.append(mse1)
    e_mse2_list.append(mse2)
    e_ssim1_list.append(ssim1)
    e_ssim2_list.append(ssim2)
    to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                              epoch=epoch+1,
                                              fmse_str=" ".join(["{}".format(mse2)]),
                                              nmse_str=" ".join(["{}".format(mse1)]),
                                              nssim_str=" ".join(["{}".format(ssim1)]),
                                              fssim_str= " ".join(["{}".format(ssim2)]),
                                              npsnr_str=" ".join(["{}".format(psnr1)]),
                                              fpsnr_str= " ".join(["{}".format(psnr2)]))
    with open(train_log_filepath, "a") as f:
        f.write(to_write)
    # if ssim1 > best_nssim and ssim2 > best_fssim:
    #     best_nssim = ssim1
    #     best_fssim = ssim2
    #     save_all_models()
    #     print("model saved update !")
    save_all_models(epoch)
    print("model saved !")
    print("[E: %d/%d]   mse1: %f, mse2: %f,ssim1:%f, ssim2:%f, psnr1:%f, psnr2:%f" % (
        epoch+1, n_epoch, mse1, mse2, ssim1, ssim2, psnr1, psnr2))
# save_all_models()