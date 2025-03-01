import torch
import numpy as np
from torch.nn import functional as F
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from thop import profile
# 计算梯度惩罚项
def cal_grad_penalty(critic, real_samples, fake_samples):
    """计算critic的惩罚项"""
    # 定义alpha
    alpha = torch.Tensor(np.random.randn(real_samples.size(0), 1, 1, 1)).cuda()

    # 从真实数据和生成数据中的连线采样
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).cuda()
    d_interpolates = critic(interpolates)  # 输出维度：[B, 1]


    fake = torch.autograd.Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = fake.cuda()

    # 对采样数据进行求导
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # 返回一个元组(value, )

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean().cuda()

    return gradient_penalty
# nearest neighbor quantization
def quantizer(w, L, device):
    top = (L-1)//2
    down = top - L + 1

    [B, W, H, C] = w.shape
    centers = torch.range(down, top).type(torch.FloatTensor).to(device)

    centers_stack = centers.reshape(1, 1, 1, 1, L)
    centers_stack = centers_stack.repeat(B, W, H, C, 1)
    # Partition W into the Voronoi tesellation over the centers
    w = w.reshape(B, W, H, C, 1)
    w_stack = w.repeat(1, 1, 1, 1, L)

    w_hard = torch.argmin(torch.abs(w_stack - centers_stack), dim=-1) + down #hard quantization

    smx = F.softmax(1.0 / torch.abs(w_stack - centers_stack + 10e-7), dim=-1)

    # Contract last dimension
    w_soft = torch.einsum('ijklm,m->ijkl', smx, centers)  # w_soft = tf.tensordot(smx, centers, axes=((-1),(0)))

    # Treat quantization as differentiable for optimization
    w_detch = (w_hard - w_soft).clone().detach().requires_grad_(False)
    w_bar = torch.round(w_detch + w_soft)

    return w_bar
# 产生符合gumble分布的随机变量
def inverse_gumbel_cdf(y, mu, beta):
    return mu - beta * torch.log(-torch.log(y))

# 使用gumble-softmax方法进行采样
def gumbel_softmax_sampling(h, mu=0, beta=1, tau=0.001):
    """
    h : (N x K) tensor. Assume we need to sample a NxK tensor, each row is an independent r.v.
    """
    shape_h = h.shape
    p = F.softmax(h, dim=1)
    y = torch.rand(shape_h).cuda() + 1e-25  # ensure all y is positive.
    g = inverse_gumbel_cdf(y, mu, beta)
    x = torch.log(p) + g  # samples follow Gumbel distribution.
    # using softmax to generate one_hot vector:
    x = x/tau
    x = F.softmax(x, dim=1)  # now, the x approximates a one_hot vector.
    return x
# AWGN信道
def awgn_channel(x, snr):
    pow = torch.mean(torch.square(x))
    std = torch.sqrt(pow * 10 ** (-snr / 10))
    noise = std * torch.randn(x.size()).cuda()
    return x + noise

# 计算PSNR
def cal_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生下三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

def create_masks(src, trg, padding_idx):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)

    return src_mask.to(device), combined_mask.to(device)


class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return (words)

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1  # 1-gram weights
        self.w2 = w2  # 2-grams weights
        self.w3 = w3  # 3-grams weights
        self.w4 = w4  # 4-grams weights

    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2,
                                       weights=(self.w1, self.w2, self.w3, self.w4)))
        return score


def greedy_decode(model, src, Rx_sig, max_len, padding_idx, start_symbol):
    """
    这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
    """
    device = torch.device('cuda')

    memory = model.channel_decoder(Rx_sig)
    macs, params = profile(model.channel_decoder, inputs=(Rx_sig))
    # print("6, macs\params:",macs,params)
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
        #        print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        # macs, params = profile(model.decoder, inputs=(outputs, memory, combined_mask, None))
        # print("7, macs\params:",macs,params)
        # print(dec_output.size())
        pred = model.dense(dec_output)
        # macs, params = profile(model.dense, inputs=(dec_output,))
        # print("8, macs\params:",macs,params)        
        # predict the word
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)
        # prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim=-1)
        # next_word = next_word.unsqueeze(1)

        # next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs

def fine_tune(f_mod, f_demod, n_mod, n_demod, n_bits, f_bits, n_rou, f_rou, n_snr, f_snr):

    # 调制
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
    f_bits_n_esti = f_demod(y_n)
    f_symbol_esti = f_mod(f_bits_n_esti)
    f_symbol_esti = f_symbol_esti / torch.sqrt(torch.mean(torch.square(f_symbol_esti)))
    res_signal = y_n - np.sqrt(f_rou) * f_symbol_esti
    n_bits_esti = n_demod(res_signal)
    # 恢复原来的长度
    n_bits_esti1 = n_bits_esti[0:n_len, :]
    f_bits_esti1 = f_bits_esti[0:f_len, :]

    return n_bits_esti1, f_bits_esti1