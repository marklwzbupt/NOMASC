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
import commpy as cpy
import commpy.modulation as mod
from minst_models import Encoder, Decoder, Multi_Discriminator
from cifar_models import Cifar_Encoder, Cifar_Decoder, Cifar_Multi_Discriminator
from SSIM import SSIM
from modulator_models import Modulator, DeModulator
import datetime
import os
import time
import torchvision.transforms as transforms
import torchvision
# parameters setting, user1: near user, user2: remote user
device = torch.device('cuda')
# 功率分配系数
n_rou = 0.3
f_rou = 1 - n_rou
#训练信噪比
n_snr = 14
f_snr = 6
c = 16
batch_size = 16
n_epoch = 200
ssim_fn = SSIM()
lambda_gp = 0.1
# model initialization
n_encoder = Cifar_Encoder(c).to(device)
f_encoder = Cifar_Encoder(c).to(device)
n_decoder = Cifar_Decoder(c).to(device)
f_decoder = Cifar_Decoder(c).to(device)
n_Multi_dis = Cifar_Multi_Discriminator().to(device)
f_Multi_dis = Cifar_Multi_Discriminator().to(device)
n_mod = Modulator(2).to(device)
f_mod = Modulator(2).to(device)
n_demod = DeModulator(2, 2).to(device)
f_demod = DeModulator(1, 2).to(device)
# 训练保存文件夹以及日志
current_time = time.strftime("%Y%m%d-%H%M%S")
result_dir = "../models/cifar_cifar/checkpoint_" + current_time
os.mkdir(result_dir)
train_log_filename = "train_log.txt"
train_log_filepath = os.path.join(result_dir, train_log_filename)
train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [f_mse] {fmse_str} [n_mse] {nmse_str} [n_ssim] {nssim_str} [f_ssim] {fssim_str}\n"

# model saving function
def save_all_models():
    n_encoder_save = result_dir + "/n_encoder_" + str(c) + "_.pkl"
    f_encoder_save = result_dir + "/f_encoder_" + str(c) + "_.pkl"
    n_decoder_save = result_dir + "/n_decoder_" + str(c) + "_.pkl"
    f_decoder_save = result_dir + "/f_decoder_" + str(c) + "_.pkl"
    n_Multi_dis_save = result_dir + "/n_md.pkl"
    f_Multi_dis_save = result_dir + "/f_md.pkl"
    torch.save(n_encoder.state_dict(), n_encoder_save)
    torch.save(n_decoder.state_dict(), n_decoder_save)
    torch.save(f_encoder.state_dict(), f_encoder_save)
    torch.save(f_decoder.state_dict(), f_decoder_save)
    torch.save(n_Multi_dis.state_dict(), n_Multi_dis_save)
    torch.save(f_Multi_dis.state_dict(), f_Multi_dis_save)

# loading dataset minst
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


# optimizer setting
opt_tran = torch.optim.Adam(
    [{'params': n_encoder.parameters(), 'lr': 0.0001},
     {'params': f_encoder.parameters(), 'lr': 0.0001},
     {'params': n_decoder.parameters(), 'lr': 0.0001},
     {'params': f_decoder.parameters(), 'lr': 0.0001}
     ])

MD1_optimizer = torch.optim.Adam(n_Multi_dis.parameters(), lr = 0.0001)
MD2_optimizer = torch.optim.Adam(f_Multi_dis.parameters(), lr = 0.0001)
# loss 列表
e_mse1_list = []
e_mse2_list = []
e_ssim1_list = []
e_ssim2_list = []
# mse损失函数
mse_fn = nn.MSELoss()

#加载pre-trained 调制解调器
n_mod.load_state_dict(torch.load("../models/pretrained_mod/n_mod_sep.pkl"))
n_demod.load_state_dict(torch.load("../models/pretrained_mod/n_demod_sep.pkl"))
f_mod.load_state_dict(torch.load("../models/pretrained_mod/f_mod_sep.pkl"))
f_demod.load_state_dict(torch.load("../models/pretrained_mod/f_demod_sep.pkl"))

best_fssim = 0.2
best_nssim = 0.2
# 训练过程
for epoch in range(n_epoch):
    #dataloader_iterator = iter(minstloader)
    mse1_list = []
    mse2_list = []
    ssim1_list = []
    ssim2_list = []
    for i_batch, data1 in enumerate(cifarloader):
        # 一个batch的图像
        images, labels1 = data1
        image1 = images[0: batch_size //2, :]
        image2 = images[batch_size //2 : batch_size, :]
        image1 = image1.reshape(image1.shape[0], 3, 32, 32)
        image2 = image2.reshape(image2.shape[0], 3, 32, 32)
        n_image = image2
        f_image = image1
        # 将图像转为float类型tensor，并加载到cuda
        n_image = n_image.type(torch.FloatTensor).cuda()
        f_image = f_image.type(torch.FloatTensor).cuda()
        # 使用语义编码器对图像进行编码
        n_semantic_coding = n_encoder(n_image) * 5 + 1
        f_semantic_coding = f_encoder(f_image) * 5 + 1
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

        # ssim损失
        image1_ssim_loss = 1 - ssim_fn(n_image, n_recon_image)
        image2_ssim_loss = 1 - ssim_fn(f_image, f_recon_image)
        ssim_loss = image1_ssim_loss + image2_ssim_loss
        ssim1_list.append(ssim_fn(n_image, n_recon_image).detach().cpu().numpy())
        ssim2_list.append(ssim_fn(f_image, f_recon_image).detach().cpu().numpy())
        "patch loss 图像真实性判别器 判别器需要识别重建图像和原始图像，生成器与之对抗，尽量缩小重建图像与原始图像的差别，不让判别器识别"
        # 将原图像输入到判别器中
        real_output1 = n_Multi_dis(n_image)
        # 将重建图像输入到判别器中，这里detach了
        fake_output_d1 = n_Multi_dis(n_recon_image.detach())
        # 将重建图像输入到判别器中，这里没有detach
        fake_output_g1 = n_Multi_dis(n_recon_image)

        # 将原图像输入到判别器中
        real_output2 = f_Multi_dis(f_image)
        # 将重建图像输入到判别器中，这里detach了
        fake_output_d2 = f_Multi_dis(f_recon_image.detach())
        # 将重建图像输入到判别器中，这里没有detach
        fake_output_g2 = f_Multi_dis(f_recon_image)

        "WGAN-GP loss"
        # generator的loss计算，也就是生成图像的平均值取负数
        g_loss1 = -torch.mean(fake_output_g1)
        g_loss2 = -torch.mean(fake_output_g2)
        g_loss = g_loss1 + g_loss2
        # 梯度惩罚项
        grad_penalty1 = cal_grad_penalty(n_Multi_dis, n_image.data, n_recon_image.data)
        grad_penalty2 = cal_grad_penalty(f_Multi_dis, f_image.data, f_recon_image.data)
        # 判别器的loss
        d_loss1 = -torch.mean(real_output1) + torch.mean(
            fake_output_d1) + lambda_gp * grad_penalty1  # 改进2、生成器和判别器的loss不取log
        d_loss2 = -torch.mean(real_output2) + torch.mean(
            fake_output_d2) + lambda_gp * grad_penalty2

        "Update"
        # 更新编码器和生成器的参数
        ae_loss = L2 + 0.01 * ssim_loss + 0.0001 * g_loss
        opt_tran.zero_grad()
        ae_loss.backward()
        opt_tran.step()

        # 更新判别器的权重
        "train patch D"
        md1_loss = d_loss1 * 0.1
        MD1_optimizer.zero_grad()
        md1_loss.backward()
        MD1_optimizer.step()

        md2_loss = d_loss2 * 0.1
        MD2_optimizer.zero_grad()
        md2_loss.backward()
        MD2_optimizer.step()

        perce_loss = 0
        
    mse1 = np.mean(np.array(mse1_list))
    mse2 = np.mean(np.array(mse2_list))
    ssim1 = np.mean(np.array(ssim1_list))
    ssim2 = np.mean(np.array(ssim2_list))
    e_mse1_list.append(mse1)
    e_mse2_list.append(mse2)
    e_ssim1_list.append(ssim1)
    e_ssim2_list.append(ssim2)
    to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                              epoch=epoch+1,
                                              fmse_str=" ".join(["{}".format(mse2)]),
                                              nmse_str=" ".join(["{}".format(mse1)]),
                                              nssim_str=" ".join(["{}".format(ssim1)]),
                                              fssim_str= " ".join(["{}".format(ssim2)]))
    with open(train_log_filepath, "a") as f:
        f.write(to_write)
    if ssim1 > best_nssim and ssim2 > best_fssim:
        best_nssim = ssim1
        best_fssim = ssim2
        save_all_models()
        print("model saved update !")
    print("[E: %d/%d]   mse1: %f, mse2: %f,ssim1:%f, ssim2:%f" % (
        epoch+1, n_epoch, mse1, mse2, ssim1, ssim2))
    print("")
# save_all_models()