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
import torchvision.transforms as transforms
import torchvision
import matlab.engine

# parameters setting, user1: near user, user2: remote user
device = torch.device('cuda')
nk = 1
fk = 1
n_rou = 0.3
f_rou = 1 - n_rou
n_snr = 14
f_snr = 6
c1 = 4
c2 = 16
batch_size1 = 32
batch_size2 = 8
n_epoch = 200
ssim_fn = SSIM()
lambda_gp = 0.1


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

eng = matlab.engine.start_matlab()
snr_list = [-8, -4, -1, 2, 5, 8, 11, 14, 17, 20]
#snr_list = [100, 200, 300, 400, 500]
ssim1_list = []
psnr1_list = []
ssim2_list = []
psnr2_list = []
ber1_list = []
ber2_list = []
quanlity =60
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
        for count in range(len(cifar_dataset.data)):
            n_snr = snr + 8
            f_snr = snr
            random1 = np.random.randint(0, len(cifar_dataset))
            random2 = np.random.randint(0, len(images))
            f_image = cifar_dataset.data[random1]
            n_image = images[random2]
            f_image = f_image.reshape(32, 32, 3)
            n_image = n_image.reshape(28, 28, 1)
            jpeg_coding_n = cv2.imencode('.jpg', n_image, [int(cv2.IMWRITE_JPEG_QUALITY), quanlity])[1]
            jpeg_coding_v = cv2.imencode('.jpg', f_image, [int(cv2.IMWRITE_JPEG_QUALITY), 60])[1]
            jpeg_bits_n = np.unpackbits(jpeg_coding_n)
            jpeg_bits_v = np.unpackbits(jpeg_coding_v)
            n_bit_esti, f_bit_esti, time = eng.ldpc_modulate_noma_awgn(jpeg_bits_n.tolist(), jpeg_bits_v.tolist(), n_rou,
                                                                 f_rou, n_snr, f_snr, nargout=3)
            # print(time)
            n_bit_esti = np.array(n_bit_esti).astype(int)
            f_bit_esti = np.array(f_bit_esti).astype(int)
            n_jpeg_esti = np.packbits(n_bit_esti)
            f_jpeg_esti = np.packbits(f_bit_esti)
            n_image_esti = cv2.imdecode(n_jpeg_esti, cv2.IMREAD_GRAYSCALE)
            f_image_esti = cv2.imdecode(f_jpeg_esti, cv2.IMREAD_COLOR)
            if n_image_esti is not None:
                n_image = torch.from_numpy(n_image).cuda().unsqueeze(0).reshape(1, np.shape(n_image)[2],
                                                                                np.shape(n_image)[0],
                                                                                np.shape(n_image)[1]) / 255
                n_image_esti = torch.from_numpy(n_image_esti).cuda().unsqueeze(0).unsqueeze(0) / 255
                ssim1 = ssim_fn(n_image, n_image_esti)
                psnr1 = cal_psnr(n_image, n_image_esti)
                ssim1_all += ssim1
                psnr1_all += psnr1
            if f_image_esti is not None:
                f_image = torch.from_numpy(f_image).cuda().unsqueeze(0).reshape(1, np.shape(f_image)[2],
                                                                                np.shape(f_image)[0],
                                                                                np.shape(f_image)[1]) / 255
                f_image_esti = torch.from_numpy(f_image_esti).cuda().unsqueeze(2).unsqueeze(0).reshape(1, np.shape(
                    f_image_esti)[2], np.shape(f_image_esti)[0], np.shape(f_image_esti)[1]) / 255
                ssim2 = ssim_fn(f_image, f_image_esti)
                psnr2 = cal_psnr(f_image, f_image_esti)
                ssim2_all += ssim2
                psnr2_all += psnr2
            cnt += 1
            #all_time += time4-time3 + time2-time1
            if cnt>20:
                ssim1_all = ssim1_all / cnt
                ssim2_all = ssim2_all / cnt
                psnr1_all = psnr1_all / cnt
                psnr2_all = psnr2_all / cnt


                # #print(all_time)
                # plt.figure()
                # #plt.title("snr:%f"%(snr))
                # plt.subplot(521)
                # plt.title("Usr_n original1")
                # plt.imshow(np.transpose(n_image[0].detach().cpu().numpy(), (1, 2, 0)))
                # plt.subplot(523)
                # plt.title("Usr_n original2")
                # plt.imshow(np.transpose(n_image[1].detach().cpu().numpy(), (1, 2, 0)))
                # plt.subplot(525)
                # plt.title("Usr_n original3")
                # plt.imshow(np.transpose(n_image[2].detach().cpu().numpy(), (1, 2, 0)))
                # plt.subplot(527)
                # plt.title("Usr_n original4")
                # plt.imshow(np.transpose(n_image[3].detach().cpu().numpy(), (1, 2, 0)))
                # plt.subplot(529)
                # plt.title("Usr_f original")
                # plt.imshow(np.transpose(f_image[0].detach().cpu().numpy(), (1, 2, 0)))
                # plt.subplot(522)
                # plt.title("Usr_n reconstruct1")
                # plt.imshow(np.transpose(n_recon_image[0].detach().cpu().numpy(), (1, 2, 0)))
                # plt.subplot(524)
                # plt.title("Usr_n reconstruct2")
                # plt.imshow(np.transpose(n_recon_image[1].detach().cpu().numpy(), (1, 2, 0)))
                # plt.subplot(526)
                # plt.title("Usr_n reconstruct3")
                # plt.imshow(np.transpose(n_recon_image[2].detach().cpu().numpy(), (1, 2, 0)))
                # plt.subplot(528)
                # plt.title("Usr_n reconstruct4")
                # plt.imshow(np.transpose(n_recon_image[3].detach().cpu().numpy(), (1, 2, 0)))
                # plt.subplot(5, 2, 10)
                # plt.title("Usr_f reconstruct")
                # plt.imshow(np.transpose(f_recon_image[0].detach().cpu().numpy(), (1, 2, 0)))
                # plt.suptitle("SNR:%f" % (snr))
                # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                #                     wspace=0, hspace=1)
                # plt.show()
                break
        if ssim1_all == 0:
            ssim1_list.append(ssim1_all)
        else:
            ssim1_list.append(ssim1_all.item())
        if ssim2_all == 0:
            ssim2_list.append(ssim2_all)
        else:
            ssim2_list.append(ssim2_all.item())
        if psnr1_all == 0:
            psnr1_list.append(psnr1_all)
        else:
            psnr1_list.append(psnr1_all.item())
        if psnr2_all == 0:
            psnr2_list.append(psnr2_all)
        else:
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



