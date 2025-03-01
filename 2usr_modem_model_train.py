import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import math
from thop import profile
from mpl_toolkits.mplot3d import Axes3D


class Encoder(torch.nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.l1 = torch.nn.Linear(1, dim)

    def forward(self, x):
        out = self.l1(x)
        return out

class Decoder(torch.nn.Module):
    def __init__(self, k, dim):
        super(Decoder, self).__init__()
        self.l1 = torch.nn.Linear(dim, 128)
        self.l2 = torch.nn.Linear(128, 64)
        self.l3 = torch.nn.Linear(64, 32)
        self.l4 = torch.nn.Linear(32, k)


    def forward(self, x):
        out1 = self.l1(x)
        out1 = F.tanh(out1)
        out1 = self.l2(out1)
        out1 = F.tanh(out1)
        out1 = self.l3(out1)
        out1 = F.tanh(out1)
        out1 = self.l4(out1)
        #out1 = F.tanh(out1)

        return out1

def awgn_channel(x, snr):
    pow = torch.mean(torch.square(x))
    std = torch.sqrt(pow * 10 ** (-snr / 10))
    noise = std * torch.randn(x.size()).cuda()
    return x + noise
def awgn_channel_given_power(x, snr, pow):
    pow = torch.tensor(pow)
    std = torch.sqrt(pow * 10 ** (-snr / 10))
    noise = std * torch.randn(x.size()).cuda()
    return x + noise
def rayleigh_fading_channel(input, snr):
    device = torch.device('cuda')
    origin_shape = input.shape
    input_reshape = input.reshape(input.shape[0], -1, 2)
    input_com = torch.complex(input_reshape[:, :, 0], input_reshape[:, :, 1]).to(device)
    [batch_size, length] = input_com.shape
    coding_shape = input_com.shape
    h = torch.complex(torch.randn(size=[1])*(1 / np.sqrt(2)), torch.randn(size=[1])*(1 / np.sqrt(2))).to(device)
    y_h = h * input_com
    power = torch.sum(input_com * torch.conj(input_com)) / (batch_size*length)

    noise_std = torch.sqrt(10 ** (-snr / 10) * power)

    awgn = torch.complex(
        torch.randn(coding_shape) * (1 / np.sqrt(2)),
        torch.randn(coding_shape) * (1 / np.sqrt(2))
    ).to(device)
    y_add = y_h + awgn * noise_std
    y_add = y_add/h
    output = torch.zeros(input_reshape.shape).to(input.device)
    output[:, :, 0] = torch.real(y_add)
    output[:, :, 1] = torch.imag(y_add)

    output = output.reshape(origin_shape)

    return output


def Rayleigh(Tx_sig, snr):
    shape = Tx_sig.shape
    H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
    H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
    H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
    Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
    Rx_sig = awgn_channel(Tx_sig, snr)
    # Channel estimation
    Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

    return Rx_sig
chan_type = "awgn"
# chan_type = "rayleigh"
device = torch.device('cuda')
# 输出维度
dim = 2
# # n-user解码所有用户的信息
n_encoder = Encoder(dim).to(device)
f_encoder = Encoder(dim).to(device)
n_decoder = Decoder(2, dim).to(device)
# f-user只需要检测自己的信息
f_decoder = Decoder(1, dim).to(device)

# 训练集大小
vol = 10000

def get_quan_prob(m):
    prob_set = 1 / 2**m * np.ones((2**m))
    if m == 1:
        quan_set = [ 0., 10.]
    elif m == 2:
        quan_set = [-3.3333,  0.0000,  3.3333,  6.6667]
    elif m == 3:
        quan_set = [-4.2857, -2.8571, -1.4286,  0.0000,  1.4286,  2.8571,  4.2857,  5.7143]
    elif m == 4:
        quan_set = [-4.0000, -3.3333, -2.6667, -2.0000, -1.3333, -0.6667,  0.0000,  0.6667,
            1.3333,  2.0000,  2.6667,  3.3333,  4.0000,  4.6667,  5.3333,  6.0000]
    elif m == 5:
        quan_set = [-3.8710, -3.5484, -3.2258, -2.9032, -2.5806, -2.2581, -1.9355, -1.6129,
            -1.2903, -0.9677, -0.6452, -0.3226,  0.0000,  0.3226,  0.6452,  0.9677,
            1.2903,  1.6129,  1.9355,  2.2581,  2.5806,  2.9032,  3.2258,  3.5484,
            3.8710,  4.1935,  4.5161,  4.8387,  5.1613,  5.4839,  5.8065,  6.1290]
    elif m == 6:
        quan_set = [-3.9683, -3.8095, -3.6508, -3.4921, -3.3333, -3.1746, -3.0159, -2.8571,
            -2.6984, -2.5397, -2.3810, -2.2222, -2.0635, -1.9048, -1.7460, -1.5873,
            -1.4286, -1.2698, -1.1111, -0.9524, -0.7937, -0.6349, -0.4762, -0.3175,
            -0.1587,  0.0000,  0.1587,  0.3175,  0.4762,  0.6349,  0.7937,  0.9524,
            1.1111,  1.2698,  1.4286,  1.5873,  1.7460,  1.9048,  2.0635,  2.2222,
            2.3810,  2.5397,  2.6984,  2.8571,  3.0159,  3.1746,  3.3333,  3.4921,
            3.6508,  3.8095,  3.9683,  4.1270,  4.2857,  4.4444,  4.6032,  4.7619,
            4.9206,  5.0794,  5.2381,  5.3968,  5.5556,  5.7143,  5.8730,  6.0317]

    return quan_set, prob_set
# 量化阶数
m_n = 2
m_f = 2
save_model = True
n_encoder_save = "models/pretrained_mod/n_mod_sep" + ".pkl"
n_decoder_save = "models/pretrained_mod/n_demod_sep"  + ".pkl"
f_encoder_save = "models/pretrained_mod/f_mod_sep" + ".pkl"
f_decoder_save = "models/pretrained_mod/f_demod_sep"  + ".pkl"
quan_n, prob_n = get_quan_prob(m_n)
quan_f, prob_f = get_quan_prob(m_f)

input_bits1 = np.random.choice(quan_n, vol, p = prob_n)
input_bits2 = np.random.choice(quan_f, vol, p = prob_f)

input_bits1 = torch.from_numpy(input_bits1).type(torch.FloatTensor).unsqueeze(1) / np.max(np.array(quan_n))
input_bits2 = torch.from_numpy(input_bits2).type(torch.FloatTensor).unsqueeze(1) / np.max(np.array(quan_f))

all_user_data = torch.cat([input_bits1, input_bits2], dim=1)

dataloader = torch.utils.data.DataLoader(all_user_data,  batch_size=vol)
# near user 的信道SNR
snr_n = 14
# far user 的信道SNR
snr_f = 6
num_epoch = 2000
t_loss1_list = []
t_loss2_list = []
# opt1 优化near user的编解码器
opt1 = torch.optim.SGD([
    {'params': n_encoder.parameters(), 'lr': 0.1},
    {'params': n_decoder.parameters(), 'lr': 0.1}])

# opt2 优化far user的编解码器
opt2 = torch.optim.SGD([
    {'params': f_encoder.parameters(), 'lr': 0.1},
    {'params': f_decoder.parameters(), 'lr': 0.1}])


mse = torch.nn.MSELoss()
# 能量分配系数
rou_n = 0.3
rou_f = 0.7
for epoch in range(num_epoch):
    loss_n_list = []
    loss_f_list = []
    error1 = 0
    error2 = 0
    for i_batch, data in enumerate(dataloader):
        data = data.to(device)
        input_n = data[:,0].to(device).unsqueeze(-1)
        input_f = data[:,1].to(device).unsqueeze(-1)

        encoded_data_n = n_encoder(input_n)
        encoded_data_f = f_encoder(input_f)
        # 归一化能量
        encoded_data_n = encoded_data_n / torch.sqrt(torch.mean(torch.square(encoded_data_n)))
        encoded_data_f = encoded_data_f / torch.sqrt(torch.mean(torch.square(encoded_data_f)))
        # superimposed
        superimposed = np.sqrt(rou_n) * encoded_data_n  + np.sqrt(rou_f) * encoded_data_f 
        # 经过信道， 用户1是near user，分配较少的能量，用户2是remote user，分配较多的能量
        if chan_type == "awgn":
            y_n = awgn_channel(superimposed, snr_n)
            y_f = awgn_channel(superimposed, snr_f)
        else:
            y_n = Rayleigh(superimposed, snr_n)
            y_f = Rayleigh(superimposed, snr_f)

        loss_n = mse(n_decoder(y_n),  data)
        loss_f = mse(f_decoder(y_f) , input_f)

        opt2.zero_grad()
        loss_f.backward(retain_graph=True)
        opt2.step()
        
        opt1.zero_grad()
        loss_n.backward()
        opt1.step()

        loss_n_list.append(mse(n_decoder(y_n)[:,0].unsqueeze(-1), input_n).detach().cpu().numpy())
        loss_f_list.append(mse(f_decoder(y_f), input_f).detach().cpu().numpy())

        loss_n_list.append(loss_n.detach().cpu().numpy())
        loss_f_list.append(loss_f.detach().cpu().numpy())

    print("Epoch:"+str(epoch)+", loss_n: "+ str(np.mean(np.array(loss_n_list))) + ", loss_f: "+ str(np.mean(np.array(loss_f_list))))

if save_model:
    torch.save(n_encoder.state_dict(), n_encoder_save)
    torch.save(n_decoder.state_dict(), n_decoder_save)
    torch.save(f_encoder.state_dict(), f_encoder_save)
    torch.save(f_decoder.state_dict(), f_decoder_save)
