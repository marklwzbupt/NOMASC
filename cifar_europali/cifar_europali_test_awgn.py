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
from thop import profile
from ptflops import get_model_complexity_info
from text_model import DeepSC
import json
from fvcore.nn import FlopCountAnalysis, parameter_count_table
device = torch.device('cuda')

n_rou = 0.3
f_rou = 1 - n_rou
n_snr = 14
f_snr = 6
batch_size_cifar = 32
batch_size_euro = 8

num_epoch = 200
ssim_fn = SSIM()
lambda_gp = 0.1
# minst，网络最后一层的通道数目
c = 16
# minst的编码器和解码器
f_encoder = Cifar_Encoder(c).to(device)
f_decoder = Cifar_Decoder(c).to(device)
f_Multi_dis = Cifar_Multi_Discriminator().to(device)
n_mod = Modulator(2).to(device)
f_mod = Modulator(2).to(device)
n_demod = DeModulator(2, 2).to(device)
f_demod = DeModulator(1, 2).to(device)
# 文本模型参数
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
vocab_file = r'E:\dataset\europarl\vocab.json'
# vocab_file = "/home/xvxiaodong/lwz/dataset/europarl/vocab.json"
""" preparing the dataset """
vocab = json.load(open(vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']
idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
num_vocab = len(token_to_idx)
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx = token_to_idx["<END>"]
StoT = SeqtoText(token_to_idx, end_idx)
# 文本模型deepSC
deepsc = DeepSC(num_layers, num_vocab, num_vocab,
                num_vocab, num_vocab, d_model, num_heads,
                dff, 0.1).to(device)
initNetParams(deepsc)
def load_all_models(current_time):
    f_encoder_save = "../models/cifar_europali/checkpoint_" +str(current_time) + "/f_encoder_" + str(c) + "_.pkl"
    f_decoder_save = "../models/cifar_europali/checkpoint_" +str(current_time) + "/f_decoder_" + str(c) + "_.pkl"
    deepsc_save = "../models/cifar_europali/checkpoint_" + str(current_time) + "/deepsc.pkl"

    f_encoder.load_state_dict(torch.load(f_encoder_save))
    f_decoder.load_state_dict(torch.load(f_decoder_save))
    deepsc.load_state_dict(torch.load(deepsc_save))

    print('model load!')

n_mod.load_state_dict(torch.load("../models/pretrained_mod/n_mod_sep.pkl"))
n_demod.load_state_dict(torch.load("../models/pretrained_mod/n_demod_sep.pkl"))
f_mod.load_state_dict(torch.load("../models/pretrained_mod/f_mod_sep.pkl"))
f_demod.load_state_dict(torch.load("../models/pretrained_mod/f_demod_sep.pkl"))
load_all_models("20221114-233838")
# 加载cifar数据集
transform = transforms.Compose(
            [transforms.ToTensor()])
cifar_dataset = torchvision.datasets.CIFAR10(root = 'E:\dataset', train = True, download = False, transform=transform)

# cifar_dataset = torchvision.datasets.CIFAR10(root = '/home/xvxiaodong/lwz/', train = True, download = False, transform=transform)
cifar_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size_cifar, shuffle=True, num_workers=0)

root =r'E:/dataset/europarl'
# root = "/home/xvxiaodong/lwz/dataset/europarl"
#euro_pali 数据集
train_eur = EurDataset(root, 'test')
# train_eur = train_eur[0:batch_size_euro * 2000]
europali_loader = DataLoader(train_eur, batch_size=batch_size_euro, num_workers=0,
                            pin_memory=True, collate_fn=collate_data)
# mse损失函数
mse_fn = nn.MSELoss()
# bleu 度量 (n-gram)
bleu_score_1gram = BleuScore(1, 0, 0, 0)
f_encoder.eval()
f_decoder.eval()
deepsc.eval()
n_mod.eval()
n_demod.eval()
f_mod.eval()
f_demod.eval()
# 在不同的SNR下进行测试
snr_list = [-8, -4, -1, 2, 5, 8, 11, 14, 17, 20]
ssim_list = []
psnr_list = []
score = []
with torch.no_grad():
    Tx_word = []
    Rx_word = []
    for snr in snr_list:
        cnt = 0
        ssim_all = 0
        psnr_all = 0
        n_snr = snr + 8
        f_snr = snr
        noise_std = SNR_to_noise(n_snr)
        word = []
        target_word = []
        dataloader_iterator = iter(cifar_loader)
        for i_batch, data1 in enumerate(europali_loader):
            try:
                data2 = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(cifar_loader)
                data2 = next(dataloader_iterator)
            # 一个batch的图像
            sentens = data1
            images, labels = data2
            images = images.reshape(images.shape[0], 3, 32, 32)
            f_image = images
            # 图像encode
            f_image = f_image.type(torch.FloatTensor).cuda()
            f_enc_output = f_encoder(f_image) * 5 + 1
            # 文本encode
            n_sent = sentens.to(device)
            src_mask = (n_sent == pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
            enc_output = deepsc.encoder(n_sent, src_mask)
            flops = FlopCountAnalysis(deepsc.encoder, (n_sent, src_mask))
            # print("FLOPs: ", flops.total())
            n_enc_output = deepsc.channel_encoder(enc_output) * 5 + 1


            # 量化并归一化
            f_semantic_coding_q = f_encoder.quant_constellation(f_enc_output) / 6.6667
            n_semantic_coding_q = deepsc.quant_constellation(n_enc_output) / 6.6667
            # print(torch.unique(n_semantic_coding_q, return_counts=True))
            n_bits = n_semantic_coding_q.reshape(-1, 1)
            f_bits = f_semantic_coding_q.reshape(-1, 1)
            # !!! 两个用户的向量长度不一致，需要padding 0
            n_len = len(n_bits)
            f_len = len(f_bits)
            bit_len = len(n_bits) if len(n_bits) > len(f_bits) else len(f_bits)
            if n_len < f_len:
                pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, f_len - n_len))
                n_bits1 = pad(n_bits)
                f_bits1 = f_bits
            else:
                pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, n_len - f_len))
                f_bits1 = pad(f_bits)
                n_bits1 = n_bits
            # 调制
            n_symbols = n_mod(n_bits1)
            f_symbols = f_mod(f_bits1)
            # 符号能量归一化
            n_symbols = n_symbols / torch.sqrt(torch.mean(torch.square(n_symbols)))
            f_symbols = f_symbols / torch.sqrt(torch.mean(torch.square(f_symbols)))

            # superimposed coding
            superimposed = np.sqrt(n_rou) * n_symbols + np.sqrt(f_rou) * f_symbols
            # 经过信道, 用户1是near user，分配较少的能量，信道条件好，用户2是far user，分配较多的能量，信道调教差
            y_n = awgn_channel(superimposed, n_snr)
            y_f = awgn_channel(superimposed, f_snr)
            start_time = time.clock()
            # far user 直接解码
            f_bits_esti = f_demod(y_f)
            # 恢复原来的长度
            n_bits_esti = n_demod(y_n)[:,0].unsqueeze(-1)
            n_bits_esti1 = n_bits_esti[0:n_len, :]
            f_bits_esti1 = f_bits_esti[0:f_len, :]
            # reshape 接收端估计得到的bit
            n_bits_esti2 = n_bits_esti1.reshape(n_enc_output.size())
            f_bits_esti2 = f_bits_esti1.reshape(f_enc_output.size())
            f_recon_image = f_decoder(f_bits_esti2)

            target = n_sent

            out = greedy_decode(deepsc, n_sent, n_bits_esti2, 30, pad_idx,
                                start_idx)

            sentences = out.cpu().numpy().tolist()
            result_string = list(map(StoT.sequence_to_text, sentences))
            word = word + result_string

            target_sent = target.cpu().numpy().tolist()
            result_string = list(map(StoT.sequence_to_text, target_sent))
            target_word = target_word + result_string

            ssim = ssim_fn(f_image, f_recon_image)
            psnr = cal_psnr(f_image, f_recon_image)

            cnt += 1
            ssim_all += ssim
            psnr_all += psnr

            if cnt>10:
                ssim_all = ssim_all / cnt
                psnr_all = psnr_all / cnt


                #print(all_time)
                plt.figure()
                # plt.title("snr:%f"%(snr))
                plt.subplot(121)
                plt.title("Usr_f original")
                plt.imshow(np.transpose(f_image[0].detach().cpu().numpy(), (1, 2, 0)))
                plt.subplot(122)
                plt.title("Usr_f reconstruct")
                plt.imshow(np.transpose(f_recon_image[0].detach().cpu().numpy(), (1, 2, 0)))
                plt.suptitle("SNR:%f" % (snr))
                plt.show()
                break
        Tx_word.append(target_word)
        Rx_word.append(word)
        ssim_list.append(ssim_all.item())
        psnr_list.append(psnr_all.item())
    bleu_score = []
    sim_score = []
    for sent1, sent2 in zip(Tx_word, Rx_word):
        # 1-gram
        bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2))  # 7*num_sent
        # sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
    bleu_score = np.array(bleu_score)
    bleu_score = np.mean(bleu_score, axis=1)
    score.append(bleu_score)
bleu = np.mean(np.array(score), axis=0)
print("ssim_list", ssim_list)
print("===============================================================================================")
print("psnr_list", psnr_list)
print("bleu_list", bleu)

plt.figure()
plt.plot(snr_list, ssim_list, 'o-', label = "F-user")
plt.ylabel('SSIM')
plt.xlabel('SNR')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot([ i + 8 for i in snr_list], bleu.tolist(), '>-', label = "N-user")
plt.ylabel('BLEU')
plt.xlabel('SNR')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(snr_list, psnr_list, 'o-', label = "F-user")
plt.ylabel('PSNR')
plt.xlabel('SNR')
plt.grid()
plt.legend()
plt.show()
