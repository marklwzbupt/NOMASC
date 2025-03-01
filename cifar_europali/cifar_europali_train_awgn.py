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
from utils import cal_grad_penalty, quantizer, gumbel_softmax_sampling, awgn_channel, initNetParams, create_masks, fine_tune
from modulator_models import Modulator, DeModulator
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from text_model import DeepSC
import time
import torchvision.transforms as transforms
import torchvision
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
current_time = time.strftime("%Y%m%d-%H%M%S")
result_dir = "../models/cifar_europali/checkpoint_" + current_time
os.mkdir(result_dir)
train_log_filename = "train_log.txt"
train_log_filepath = os.path.join(result_dir, train_log_filename)
# 保存的格式
train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [ce_Loss] {loss_str} [n_mse] {mse_str} [n_ssim] {ssim_str} [fmodloss] {fmodloss_str} [nmodloss] {nmodloss_str}\n"
def save_all_models():
    f_encoder_save = result_dir + "/f_encoder_" + str(c) + "_.pkl"
    f_decoder_save = result_dir + "/f_decoder_" + str(c) + "_.pkl"
    f_Multi_dis_save = result_dir + "/f_md.pkl"
    deepsc_save = result_dir  + "/deepsc.pkl"

    torch.save(f_encoder.state_dict(), f_encoder_save)
    torch.save(f_decoder.state_dict(), f_decoder_save)
    torch.save(deepsc.state_dict(), deepsc_save)
    torch.save(f_Multi_dis.state_dict(), f_Multi_dis_save)

# 文本模型参数
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
# vocab_file = r'E:\dataset\europarl\vocab.json'
vocab_file = "/home/xvxiaodong/lwz/dataset/europarl/vocab.json"
""" preparing the dataset """
vocab = json.load(open(vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']
num_vocab = len(token_to_idx)
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx = token_to_idx["<END>"]
# 文本模型deepSC
deepsc = DeepSC(num_layers, num_vocab, num_vocab,
                num_vocab, num_vocab, d_model, num_heads,
                dff, 0.1).to(device)
initNetParams(deepsc)
# 加载cifar数据集
transform = transforms.Compose(
            [transforms.ToTensor()])
# cifar_dataset = torchvision.datasets.CIFAR10(root = 'E:\dataset', train = True, download = False, transform=transform)

cifar_dataset = torchvision.datasets.CIFAR10(root = '/home/xvxiaodong/lwz/', train = True, download = False, transform=transform)
cifar_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size_cifar, shuffle=True, num_workers=0)

# root =r'E:/dataset/europarl/'
root = "/home/xvxiaodong/lwz/dataset/europarl"
#euro_pali 数据集
train_eur = EurDataset(root, 'train')
# train_eur = train_eur[0:batch_size_euro * 2000]
europali_loader = DataLoader(train_eur, batch_size=batch_size_euro, num_workers=0,
                            pin_memory=True, collate_fn=collate_data)

# optimizer setting
opt_tran = torch.optim.Adam(
    [{'params': f_encoder.parameters(), 'lr': 0.0001},
     {'params': deepsc.parameters(), 'lr': 1e-4, 'betas': (0.9, 0.98), 'eps': 1e-8, 'weight_decay': 5e-4},
     {'params': f_decoder.parameters(), 'lr': 0.0001}
    # ,{'params': n_mod.parameters(), 'lr': 1e-4},
    #  {'params': f_mod.parameters(), 'lr': 1e-4},
    #  {'params': n_demod.parameters(), 'lr': 1e-4},
    #  {'params': f_demod.parameters(), 'lr': 1e-4}
     ])

opt_fmod = torch.optim.SGD(
    [#{'params': n_mod.parameters(), 'lr': 0.1},
     {'params': f_mod.parameters(), 'lr': 0.1},
     # {'params': n_demod.parameters(), 'lr': 0.1},
     {'params': f_demod.parameters(), 'lr': 0.1}
     ])

opt_nmod = torch.optim.SGD(
    [#{'params': n_mod.parameters(), 'lr': 0.1},
     {'params': n_mod.parameters(), 'lr': 0.1},
     # {'params': n_demod.parameters(), 'lr': 0.1},
     {'params': n_demod.parameters(), 'lr': 0.1}
     ])
MD1_optimizer = torch.optim.Adam(f_Multi_dis.parameters(), lr = 0.0001)

# 损失函数设定
ce_loss = nn.CrossEntropyLoss(reduction = 'none')
# mse损失函数
mse_fn = nn.MSELoss()
# bce损失函数
bce_loss = nn.BCELoss()
# loss 列表
e_mse_list = []
e_ssim_list = []

# 调制解调器加载权重
n_mod.load_state_dict(torch.load("../models/pretrained_mod/n_mod_sep.pkl"))
n_demod.load_state_dict(torch.load("../models/pretrained_mod/n_demod_sep.pkl"))
f_mod.load_state_dict(torch.load("../models/pretrained_mod/f_mod_sep.pkl"))
f_demod.load_state_dict(torch.load("../models/pretrained_mod/f_demod_sep.pkl"))
# 训练过程
best_celoss = 4
best_ssim = 0.2
for epoch in range(num_epoch):
    dataloader_iterator = iter(cifar_loader)
    mse_list = []
    ce_list = []
    ssim_list = []
    f_mod_loss = []
    n_mod_loss = []
    for i_batch, data1 in enumerate(europali_loader):
        try:
            data2 = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(cifar_loader)
            data2 = next(dataloader_iterator)
        #with torch.autograd.set_detect_anomaly(True):
        # 一个batch的图像
        sentens = data1
        images, labels = data2
        images = images.reshape(images.shape[0], 3, 32, 32)
        f_image = images
        n_sent = sentens.to(device)
        trg_inp = n_sent[:, :-1]
        trg_real = n_sent[:, 1:]
        src_mask, look_ahead_mask = create_masks(n_sent, trg_inp, pad_idx)
        enc_output = deepsc.encoder(n_sent, src_mask)

        n_enc_output = deepsc.channel_encoder(enc_output) * 5 + 1
        # n_enc_output = n_enc_output.unsqueeze(0)
        f_image = f_image.type(torch.FloatTensor).cuda()
        f_enc_output = f_encoder(f_image) * 5 + 1

        # 将图像转为float类型tensor，并加载到cuda

        # 使用语义编码器对图像进行编码, 输出的结果视为被量化为1的概率

        f_semantic_coding_q = f_encoder.quant_constellation(f_enc_output) / 6.6667
        n_semantic_coding_q = deepsc.quant_constellation(n_enc_output) / 6.6667

        n_bits = n_semantic_coding_q.reshape(-1, 1)
        f_bits = f_semantic_coding_q.reshape(-1, 1)
        # !!! 两个用户的向量长度不一致，需要padding 0
        n_len = len(n_bits)
        f_len = len(f_bits)
        bit_len = len(n_bits) if len(n_bits) > len(f_bits) else len(f_bits)
        if n_len < f_len:
            pad = torch.nn.ZeroPad2d(padding = (0, 0, 0, f_len - n_len))
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
        n_symbols1 = n_symbols / torch.sqrt(torch.mean(torch.square(n_symbols)))
        f_symbols1 = f_symbols / torch.sqrt(torch.mean(torch.square(f_symbols)))


        # superimposed coding
        superimposed = np.sqrt(n_rou) * n_symbols1 + np.sqrt(f_rou) * f_symbols1
        # 经过信道, 用户1是near user，分配较少的能量，信道条件好，用户2是far user，分配较多的能量，信道调教差
        y_n = awgn_channel(superimposed, n_snr)
        y_f = awgn_channel(superimposed, f_snr)
        # far user 直接解码
        f_bits_esti = f_demod(y_f)
        # near user 使用sic进行解码
        n_bits_esti = n_demod(y_n)[:,0].unsqueeze(-1)
        # n_bits_esti = n_demod(y_n)
        # 恢复原来的长度
        n_bits_esti1 = n_bits_esti[0:n_len, :]
        f_bits_esti1 = f_bits_esti[0:f_len, :]
        # reshape 接收端估计得到的bit

        n_bits_esti2 = n_bits_esti1.reshape(n_enc_output.size())
        f_bits_esti2 = f_bits_esti1.reshape(f_enc_output.size())
        # 用decoder重建图像

        f_recon_image = f_decoder(f_bits_esti2)
        channel_dec_output = deepsc.channel_decoder(n_bits_esti2)
        dec_output = deepsc.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
        pred = deepsc.dense(dec_output)

        # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
        ntokens = pred.size(-1)

        n_loss = ce_loss(pred.contiguous().view(-1, ntokens), trg_real.contiguous().view(-1))
        mask = (trg_real.contiguous().view(-1) != pad_idx).type_as(n_loss.data)
        # a = mask.cpu().numpy()
        n_loss *= mask
        n_loss = n_loss.mean()

        #recon_image2 = decoder_2(superimposed)
        # L2损失
        f_image_L2 = ((f_recon_image - f_image) ** 2).mean()

        L2 = f_image_L2
        mse_list.append(f_image_L2.detach().cpu().numpy())


        # ssim损失
        image1_ssim_loss = 1 - ssim_fn(f_image, f_recon_image)

        ssim_loss = image1_ssim_loss
        ssim_list.append(ssim_fn(f_image, f_recon_image).detach().cpu().numpy())

        "patch loss 图像真实性判别器 判别器需要识别重建图像和原始图像，生成器与之对抗，尽量缩小重建图像与原始图像的差别，不让判别器识别"
        # 将原图像输入到判别器中
        real_output1 = f_Multi_dis(f_image)
        # 将重建图像输入到判别器中，这里detach了
        fake_output_d1 = f_Multi_dis(f_recon_image.detach())
        # 将重建图像输入到判别器中，这里没有detach
        fake_output_g1 = f_Multi_dis(f_recon_image)

        "WGAN-GP loss"
        # generator的loss计算，也就是生成图像的平均值取负数
        g_loss = -torch.mean(fake_output_g1)

        # 梯度惩罚项
        grad_penalty1 = cal_grad_penalty(f_Multi_dis, f_image.data, f_recon_image.data)

        # 判别器的loss
        d_loss1 = -torch.mean(real_output1) + torch.mean(
            fake_output_d1) + lambda_gp * grad_penalty1  # 改进2、生成器和判别器的loss不取log
        "Update"
        # 更新编码器和生成器的参数
        ae_loss = L2 + 0.01 * ssim_loss + 0.0001 * g_loss + n_loss
        opt_tran.zero_grad()
        ae_loss.backward()
        opt_tran.step()
        ce_list.append(n_loss.detach().cpu().numpy())

        # 更新判别器的权重
        "train patch D"
        md1_loss = d_loss1 * 0.1
        MD1_optimizer.zero_grad()
        md1_loss.backward()
        MD1_optimizer.step()


        # fine-tune modulator and demodulator
        n_bits_d = n_bits.detach()
        f_bits_d = f_bits.detach()
        n_bits_esti_d, f_bits_esti_d = fine_tune(f_mod, f_demod, n_mod, n_demod, n_bits_d, f_bits_d, n_rou, f_rou, n_snr, f_snr)
        nmod_loss = mse_fn(n_bits_d,  n_bits_esti_d) # + mse_fn(f_bits1, f_bits_esti)
        fmod_loss = mse_fn(f_bits_d, f_bits_esti_d)  # + mse_fn(f_bits1, f_bits_esti)

        f_mod_loss.append(fmod_loss.detach().cpu().numpy())
        n_mod_loss.append(nmod_loss.detach().cpu().numpy())

        perce_loss = 0
        
    mse = np.mean(np.array(mse_list))
    ssim = np.mean(np.array(ssim_list))
    celoss = np.mean(ce_list)
    fmodloss = np.mean(f_mod_loss)
    nmodloss = np.mean(n_mod_loss)
    e_mse_list.append(mse)
    e_ssim_list.append(ssim)
    to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                              epoch=epoch+1,
                                              loss_str=" ".join(["{}".format(n_loss)]),
                                              mse_str=" ".join(["{}".format(mse)]),
                                              ssim_str=" ".join(["{}".format(ssim)]),
                                              fmodloss_str = " ".join(["{}".format(fmodloss)]),
                                              nmodloss_str = " ".join(["{}".format(nmodloss)]))
    with open(train_log_filepath, "a") as f:
        f.write(to_write)
    if ssim > best_ssim and n_loss < best_celoss:
        best_ssim = ssim
        best_celoss = n_loss
        save_all_models()
        print("model saved update !")
    print("[E: %d/%d]   n_mse: %f, f_loss: %f,ssim:%f, f_modloss:%f, n_modloss:%f " % (
        epoch+1, num_epoch, mse, n_loss.item(), ssim, fmodloss, nmodloss))
    print("")
# save_all_models()