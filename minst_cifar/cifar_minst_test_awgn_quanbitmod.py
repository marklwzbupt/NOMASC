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
from utils import cal_grad_penalty, quantizer, gumbel_softmax_sampling, awgn_channel, cal_psnr
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
from utils_tradition import quan_to_bit_mappping, bit_to_quan_mapping, awgn_channel_np
import torchvision.transforms as transforms
import torchvision
# parameters setting, user1: near user, user2: remote user
device = torch.device('cuda')

n_rou = 0.3
f_rou = 1 - n_rou
n_snr = 14
f_snr = 6
c1 = 6
c2 = 24
batch_size1 = 32
batch_size2 = 8
n_epoch = 200
ssim_fn = SSIM()
lambda_gp = 0.1
# model initialization
n_encoder = Encoder(c1).to(device)
f_encoder = Cifar_Encoder(c2).to(device)
n_decoder = Decoder(c1).to(device)
f_decoder = Cifar_Decoder(c2).to(device)
n_Multi_dis = Multi_Discriminator().to(device)
f_Multi_dis = Cifar_Multi_Discriminator().to(device)

def load_all_models(current_time):
    n_encoder_save = "../models/minst_cifar/checkpoint_" +str(current_time) + "/n_encoder_" + str(c1) + "_.pkl"
    f_encoder_save = "../models/minst_cifar/checkpoint_" +str(current_time) + "/f_encoder_" + str(c2) + "_.pkl"
    n_decoder_save = "../models/minst_cifar/checkpoint_" +str(current_time) + "/n_decoder_" + str(c1) + "_.pkl"
    f_decoder_save = "../models/minst_cifar/checkpoint_" +str(current_time) + "/f_decoder_" + str(c2) + "_.pkl"

    n_encoder.load_state_dict(torch.load(n_encoder_save))
    f_encoder.load_state_dict(torch.load(f_encoder_save))
    n_decoder.load_state_dict(torch.load(n_decoder_save))
    f_decoder.load_state_dict(torch.load(f_decoder_save))
load_all_models("20221112-112945")

# loading dataset minst
#images, labels = load_mnist("/home/yqk/drl_for_vne/dataset/mnist_dataset/", kind='train')
images, labels = load_mnist(r"E:\dataset\mnist_dataset", kind='train')
# images, labels = load_mnist("/home/xvxiaodong/lwz/dataset/mnist_dataset/", kind='train')
data = MNIST(images, labels)
minstloader = DataLoader(data, batch_size=batch_size1, shuffle=True)

# load dataset cifar
transform = transforms.Compose(
            [transforms.ToTensor()])
cifar_dataset = torchvision.datasets.CIFAR10(root = 'E:\dataset', train = True, download = False, transform=transform)

# cifar_dataset = torchvision.datasets.CIFAR10(root = '/home/xvxiaodong/lwz/', train = True, download = False, transform=transform)
cifarloader = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size2, shuffle=True, num_workers=0)

# loss 列表
e_mse1_list = []
e_mse2_list = []
e_ssim1_list = []
e_ssim2_list = []
# mse损失函数
mse_fn = nn.MSELoss()
# bce损失函数
bce_loss = nn.BCELoss()
n_encoder.eval()
f_encoder.eval()
n_decoder.eval()
f_decoder.eval()


snr_list = [-7, -4, -1, 2, 5, 8, 11, 14, 17, 20]
#snr_list = [100, 200, 300, 400, 500]
ssim1_list = []
psnr1_list = []
ssim2_list = []
psnr2_list = []
ber1_list = []
ber2_list = []
modem = mod.QAMModem(4)
with torch.no_grad():
    for snr in snr_list:
        cnt = 0
        ssim1_all = 0
        ssim2_all = 0
        psnr1_all = 0
        psnr2_all = 0
        ber1_all = 0
        ber2_all = 0
        all_time = 0
        dataloader_iterator = iter(minstloader)
        for i_batch, data1 in enumerate(cifarloader):
            n_snr = snr + 8
            f_snr = snr
            try:
                data2 = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(minstloader)
                data2 = next(dataloader_iterator)

            # 一个batch的图像
            images1, labels1 = data1
            images2, labels2 = data2
            images1 = images1.reshape(images1.shape[0], 3, 32, 32)
            images2 = images2.reshape(images2.shape[0], 1, 28, 28)
            n_image = images2
            f_image = images1
            # 将图像转为float类型tensor，并加载到cuda
            n_image = n_image.type(torch.FloatTensor).cuda()
            f_image = f_image.type(torch.FloatTensor).cuda()
            # 使用语义编码器对图像进行编码, 输出的结果视为被量化为1的概率
            n_semantic_coding = n_encoder(n_image) * 5 + 1
            f_semantic_coding = f_encoder(f_image) * 5 + 1
            n_semantic_coding_q = n_encoder.quant_constellation(n_semantic_coding) / 6.6667
            f_semantic_coding_q = f_encoder.quant_constellation(f_semantic_coding) / 6.6667

            n_bits = np.array(quan_to_bit_mappping(n_semantic_coding_q.reshape(-1, 1)))
            f_bits = np.array(quan_to_bit_mappping(f_semantic_coding_q.reshape(-1, 1)))
            len1 = int(len(n_bits) / 2)
            len2 = int(len(f_bits) / 2)
            n_symbol = modem.modulate(n_bits)
            f_symbol = modem.modulate(f_bits)
            n_symbol = n_symbol / np.sqrt(np.mean(n_symbol * np.conjugate(n_symbol)))
            f_symbol = f_symbol / np.sqrt(np.mean(f_symbol * np.conjugate(f_symbol)))
            sc_coding = n_symbol * np.sqrt(n_rou) + f_symbol * np.sqrt(f_rou)
            yn = awgn_channel_np(sc_coding, n_snr)
            yf = awgn_channel_np(sc_coding, f_snr)

            f_bit_esti = modem.demodulate(yf, 'hard')
            f_quan_esti = bit_to_quan_mapping(f_bit_esti)
            f_quan_esti = f_quan_esti[0:len2]
            # sic
            f_bit_n_esti = modem.demodulate(yn, 'hard')
            strong_recon = modem.modulate(f_bit_n_esti)
            residual = yn - np.sqrt(f_rou) * strong_recon
            n_bit_esti = modem.demodulate(residual, 'hard')
            n_quan_esti = bit_to_quan_mapping(n_bit_esti)
            n_quan_esti = n_quan_esti[0:len1]
            n_bits_esti2 = torch.reshape(n_quan_esti, n_semantic_coding_q.size())
            f_bits_esti2 = torch.reshape(f_quan_esti, f_semantic_coding_q.size())

            n_recon_image = n_decoder(n_bits_esti2)
            f_recon_image = f_decoder(f_bits_esti2)
            ssim1 = ssim_fn(n_image, n_recon_image)
            psnr1 = cal_psnr(n_image, n_recon_image)
            ssim2 = ssim_fn(f_image, f_recon_image)
            psnr2 = cal_psnr(f_image, f_recon_image)



            cnt += 1
            ssim1_all += ssim1
            psnr1_all += psnr1
            ssim2_all += ssim2
            psnr2_all += psnr2

            #all_time += time4-time3 + time2-time1
            if cnt>20:
                ssim1_all = ssim1_all / cnt
                ssim2_all = ssim2_all / cnt
                psnr1_all = psnr1_all / cnt
                psnr2_all = psnr2_all / cnt


                #print(all_time)
                plt.figure()
                #plt.title("snr:%f"%(snr))
                plt.subplot(521)
                plt.title("Usr_n original1")
                plt.imshow(np.transpose(n_image[0].detach().cpu().numpy(), (1, 2, 0)))
                plt.subplot(523)
                plt.title("Usr_n original2")
                plt.imshow(np.transpose(n_image[1].detach().cpu().numpy(), (1, 2, 0)))
                plt.subplot(525)
                plt.title("Usr_n original3")
                plt.imshow(np.transpose(n_image[2].detach().cpu().numpy(), (1, 2, 0)))
                plt.subplot(527)
                plt.title("Usr_n original4")
                plt.imshow(np.transpose(n_image[3].detach().cpu().numpy(), (1, 2, 0)))
                plt.subplot(529)
                plt.title("Usr_f original")
                plt.imshow(np.transpose(f_image[0].detach().cpu().numpy(), (1, 2, 0)))
                plt.subplot(522)
                plt.title("Usr_n reconstruct1")
                plt.imshow(np.transpose(n_recon_image[0].detach().cpu().numpy(), (1, 2, 0)))
                plt.subplot(524)
                plt.title("Usr_n reconstruct2")
                plt.imshow(np.transpose(n_recon_image[1].detach().cpu().numpy(), (1, 2, 0)))
                plt.subplot(526)
                plt.title("Usr_n reconstruct3")
                plt.imshow(np.transpose(n_recon_image[2].detach().cpu().numpy(), (1, 2, 0)))
                plt.subplot(528)
                plt.title("Usr_n reconstruct4")
                plt.imshow(np.transpose(n_recon_image[3].detach().cpu().numpy(), (1, 2, 0)))
                plt.subplot(5, 2, 10)
                plt.title("Usr_f reconstruct")
                plt.imshow(np.transpose(f_recon_image[0].detach().cpu().numpy(), (1, 2, 0)))
                plt.suptitle("SNR:%f" % (snr))
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                    wspace=0, hspace=1)
                # plt.show()
                break
        ssim1_list.append(ssim1_all.item())
        ssim2_list.append(ssim2_all.item())
        psnr1_list.append(psnr1_all.item())
        psnr2_list.append(psnr2_all.item())

print("ssim1_list", ssim1_list)
print("===============================================================================================")
print("ssim2_list", ssim2_list)
print("===============================================================================================")
print("psnr1_list", psnr1_list)
print("===============================================================================================")
print("psnr2_list", psnr2_list)
print("===============================================================================================")

plt.figure()
plt.plot(snr_list, ssim2_list, 'o-', label = "F-user, ")
plt.plot([ i + 8 for i in snr_list], ssim1_list, '>-', label = "N-user")
plt.ylabel('SSIM')
plt.xlabel('SNR')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(snr_list, psnr2_list, 'o-', label = "F-user")
plt.plot([ i + 8 for i in snr_list], psnr1_list, '>-', label = "N-user")
plt.ylabel('PSNR')
plt.xlabel('SNR')
plt.grid()
plt.legend()
plt.show()
