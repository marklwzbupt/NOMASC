import os
import sys
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   #os.path.dirname通俗的讲是返回上一级文件夹绝对路径的意思，多套一层就多返回一层
sys.path.append(path)    #将路径添加到python的搜索路径中
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from europali_dataset import EurDataset, collate_data
from minst_loader import load_mnist, MNIST
from minst_models import Encoder, Decoder, Multi_Discriminator
from cifar_models import Cifar_Encoder, Cifar_Decoder, Cifar_Multi_Discriminator
from SSIM import SSIM
from utils import cal_grad_penalty, quantizer, gumbel_softmax_sampling, awgn_channel, initNetParams, create_masks, fine_tune, SeqtoText, BleuScore, SNR_to_noise,greedy_decode, cal_psnr
from modulator_models import Modulator, DeModulator
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from text_model import DeepSC
import time
import torchvision.transforms as transforms
import torchvision
import pickle
from dahuffman import HuffmanCodec
import matlab.engine
import cv2
device = torch.device('cuda')

n_rou = 0.3
f_rou = 1 - n_rou
n_snr = 14
f_snr = 6
batch_size_cifar = 32
batch_size_euro = 32
num_epoch = 200
ssim_fn = SSIM()
lambda_gp = 0.1
all_sentences_file = r'E:\dataset\europarl\all_sentences.pkl'
""" preparing the dataset """
with open(all_sentences_file, 'rb') as f:
    sentences = pickle.load(f)
all_sents = ''.join(sentences)
codec = HuffmanCodec.from_data(all_sents)
codec.print_code_table()
eng = matlab.engine.start_matlab()
word = []
target_word = []
txword = []
rxword = []
bleu_score_1gram = BleuScore(1, 0, 0, 0)
# load dataset cifar
transform = transforms.Compose(
            [transforms.ToTensor()])
cifar_dataset = torchvision.datasets.CIFAR10(root = 'E:\dataset', train = True, download = False, transform=transform)
eng = matlab.engine.start_matlab()
quanlity = 60
snr_list = [-7, -4, -1, 2, 5, 8, 11, 14, 17, 20]
#snr_list = [100, 200, 300, 400, 500]
ssim_list = []
psnr_list = []
with torch.no_grad():
    for snr in snr_list:
        cnt = 0
        ssim_all = 0
        psnr_all = 0
        all_time = 0
        word = []
        target_word = []
        for count in range(len(cifar_dataset.data)):
            print("SNR:", snr, "iteration:", cnt+1)
            n_snr = snr + 8
            f_snr = snr
            random1 = np.random.randint(0, len(cifar_dataset))
            random2 = np.random.randint(0, len(sentences))
            image = cifar_dataset.data[random1]
            sent = sentences[random2]
            f_image = image.reshape(32, 32, 3)
            jpeg_coding_f = cv2.imencode('.jpg', f_image, [int(cv2.IMWRITE_JPEG_QUALITY), quanlity])[1]
            jpeg_bits_f = np.unpackbits(jpeg_coding_f)
            huff_encoded = codec.encode(sent)
            huff_bits = []
            [huff_bits.extend([int(y) for y in '0' * (8 - len(bin(x)[2:])) + bin(x)[2:]]) for x in huff_encoded]
            huff_bits_esti, jpeg_bits_esti, time = eng.turbo_ldpc_modulate_noma_awgn(huff_bits, jpeg_bits_f.tolist(),n_rou,
                                                                                   f_rou, n_snr, f_snr, nargout=3)
            # print(time)
            f_bit_esti = np.array(jpeg_bits_esti).astype(int)
            f_jpeg_esti = np.packbits(f_bit_esti)
            f_image_esti = cv2.imdecode(f_jpeg_esti, cv2.IMREAD_COLOR)

            if f_image_esti is not None:
                f_image = torch.from_numpy(f_image).cuda().unsqueeze(0).reshape(1, np.shape(f_image)[2], np.shape(f_image)[0],
                                                                     np.shape(f_image)[1]) /255
                f_image_esti = torch.from_numpy(f_image_esti).cuda().unsqueeze(2).unsqueeze(0).reshape(1, np.shape(f_image_esti)[2], np.shape(f_image_esti)[0], np.shape(f_image_esti)[1]) / 255
                ssim = ssim_fn(f_image, f_image_esti)
                psnr = cal_psnr(f_image, f_image_esti)
                ssim_all += ssim
                psnr_all += psnr
            cnt += 1
            huff_bit_esti = np.array(huff_bits_esti).astype(int)
            huff_bytes_esti = bytes(np.packbits(huff_bit_esti))
            try:
                # decoded = bytes_esti.decode()
                decoded = codec.decode(huff_bytes_esti)
                target_word += [sent]
                word += [decoded]
            except UnicodeDecodeError:
                print("cant decode sentences")
            #all_time += time4-time3 + time2-time1
            if cnt > 20:
                ssim_all = ssim_all / cnt
                psnr_all = psnr_all / cnt

                #
                # #print(all_time)
                # plt.figure()
                # #plt.title("snr:%f"%(snr))
                # plt.subplot(221)
                # plt.title("Usr1 original")
                # plt.imshow(np.transpose(n_image[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
                # plt.subplot(222)
                # plt.title("Usr1 reconstruct")
                # plt.imshow(np.transpose(n_recon_image[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
                # plt.subplot(223)
                # plt.title("Usr2 original")
                # plt.imshow(np.transpose(f_image[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
                # plt.subplot(224)
                # plt.title("Usr2 reconstruct")
                # plt.imshow(np.transpose(f_recon_image[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
                # plt.suptitle("SNR:%f"%(snr))
                # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                #                     wspace=0, hspace=0.5)
                # plt.show()
                break
        if ssim_all == 0:
            ssim_list.append(ssim_all)
        else:
            ssim_list.append(ssim_all.item())

        if psnr_all == 0:
            psnr_list.append(psnr_all)
        else:
            psnr_list.append(psnr_all.item())

        txword.append(target_word)
        rxword.append(word)
bleu_score = []
sim_score = []
for sent1, sent2 in zip(txword, rxword):
    # 1-gram
    bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2))  # 7*num_sent
    # sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
bleu_score = np.array(bleu_score)
bleu_score1 = np.mean(bleu_score, axis=1)
print("ssim1_list", ssim_list)
print("===============================================================================================")
print("psnr1_list", psnr_list)
print("===============================================================================================")
print("bleu_list", bleu_score1)
print("===============================================================================================")


plt.figure()
plt.plot(snr_list, ssim_list, 'o-', label = "F-user, JPEG+LDPC")

plt.ylabel("SSIM of F-user")
plt.xlabel('SNR')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(snr_list, psnr_list, 'o-', label = "F-user, JPEG+LDPC")

plt.ylabel('PSNR of F-user')
plt.xlabel('SNR')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot([ i + 8 for i in snr_list], bleu_score1, 'o-', label = "N-user, Huffman+Turbo")

plt.ylabel("BLEU of N-user")
plt.xlabel('SNR')
plt.grid()
plt.legend()
plt.show()