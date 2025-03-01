import math
import sys
import numpy as np
import matplotlib.pyplot as plt


font1 = {'family': 'Times New Roman',
        'color':  'blue',
        'weight': 'normal',
        'size': 10,
        }
# def logistic_increase_function(snr, Ak1, Ak2, Ck1, Ck2):
#     # t:time   t0:initial time    P0:initial_value    K:capacity  r:increase_rate
#     ssim_p = Ak1 + (Ak2 - Ak1) / (1 + np.exp(-(Ck1*snr + Ck2)))
#     return ssim_p

# def europali_func(snr,  Ak1 = 0.36606908, Ak2 = 0.90520419, Ck1 = 0.43258331, Ck2 = -1.14715606):
#     ssim_p = Ak1 + (Ak2 - Ak1) / (1 + np.exp(-(Ck1*snr + Ck2)))
#     return ssim_p
# def cifar_func(snr,  Ak1 = 0.16843731, Ak2 = 0.85533235,Ck1 =  0.24833012, Ck2 = 0.85807546):
#     ssim_p = Ak1 + (Ak2 - Ak1) / (1 + np.exp(-(Ck1*snr + Ck2)))
#     return ssim_p

# def reverse_europali_func(ssim, Ak1 = 0.36606908, Ak2 = 0.90520419, Ck1 = 0.43258331, Ck2 = -1.14715606):
#     if (Ak2 - Ak1)/(ssim - Ak1) - 1 <= 0:
#         return False
#     else:
#         snr = (np.log((Ak2 - Ak1)/(ssim - Ak1) - 1)+Ck2)/(-Ck1)
#         return snr

# def reverse_cifar_func(ssim, Ak1 = 0.16843731, Ak2 = 0.85533235,Ck1 =  0.24833012, Ck2 = 0.85807546):
#     if (Ak2 - Ak1)/(ssim - Ak1) - 1 <= 0:
#         return False
#     else:
#         snr = (np.log((Ak2 - Ak1)/(ssim - Ak1) - 1)+Ck2)/(-Ck1)
#         return snr
def logistic_increase_function(snr, Ak1, Ak2, Ck1, Ck2):
    # t:time   t0:initial time    P0:initial_value    K:capacity  r:increase_rate
    ssim_p = Ak1 + (Ak2 - Ak1) / (1 + np.exp(-(Ck1*snr + Ck2)))
    return ssim_p
class Logistic_func:
    def __init__(self, params):
        self.Ak1 = params[0]
        self.Ak2 = params[1]
        self.Ck1 = params[2]
        self.Ck2 = params[3]
    def compute(self, x):
        ssim_p = self.Ak1 + (self.Ak2 - self.Ak1) / (1 + np.exp(-(self.Ck1*x + self.Ck2)))
        return ssim_p
    
class Reverse_logistic_func:
    def __init__(self, params):
        self.Ak1 = params[0]
        self.Ak2 = params[1]
        self.Ck1 = params[2]
        self.Ck2 = params[3]
    def compute(self, x):
        if (self.Ak2 - self.Ak1)/(x - self.Ak1) - 1 <= 0:
            return False
        else:
            snr = (np.log((self.Ak2 - self.Ak1)/(x - self.Ak1) - 1)+self.Ck2)/(-self.Ck1)
            return snr
# cifar 系数
cifar_params = [[0.16843731, 0.85533235, 0.24833012, 0.85807546],
[0.16873276, 0.92191598, 0.2631357 , 1.19284477],
[0.17735017, 0.94254289 ,0.28691992, 1.47669145],
[0.17723744 ,0.95456146, 0.2981444 , 1.68683409],
[0.18337713, 0.96069141 ,0.30777787, 1.88960563],
[0.19084881 ,0.96442812, 0.31518164 ,2.07208737]]

# europali系数
r_list = [0.08, 0.17, 0.25, 0.33, 0.42, 0.5]
europarl_params = [[0.48039465, 0.5772886,  0.26123709, 1.49508526],
[ 0.40634313 , 0.79574065 , 0.21536547, -0.35700441],
[ 0.36606908,  0.90520419 , 0.43258331 ,-1.14715606],
[0.3341525,  0.93411533, 0.47486085 ,0.17140107],
[0.40302623, 0.93330533, 0.57124942 ,1.10974694],
[0.38810838, 0.93286502, 0.60425427, 2.1793595 ]]
K = 8
r = 0.08
cifar_func = Logistic_func(cifar_params[np.searchsorted(r_list, r)])
reverse_cifar_func = Reverse_logistic_func(cifar_params[np.searchsorted(r_list, r)])
europali_func = Logistic_func(europarl_params[int(np.log2(K)-1)])
reverse_europali_func = Reverse_logistic_func(europarl_params[int(np.log2(K)-1)])
# cifar 系数
# [0.16843731 0.85533235 0.24833012 0.85807546]
# [0.16873276 0.92191598 0.2631357  1.19284477]
# [0.17735017 0.94254289 0.28691992 1.47669145]
# [0.17723744 0.95456146 0.2981444  1.68683409]
# [0.18337713 0.96069141 0.30777787 1.88960563]
# [0.19084881 0.96442812 0.31518164 2.07208737]

# europali系数
# [0.48039465 0.5772886  0.26123709 1.49508526]
# [ 0.40634313  0.79574065  0.21536547 -0.35700441]
# [0.38810838 0.93286502 0.60425427 2.1793595 ]
# [ 0.36606908  0.90520419  0.43258331 -1.14715606]
# [0.3341525  0.93411533 0.47486085 0.17140107]
# [0.40302623 0.93330533 0.57124942 1.10974694]
# 用户n是图像用户，用户f是文本用户
factor_n = 20
factor_f = 16
# epsilon_req = 0.5
# xi_req 
epsilon_req = np.arange(0.6, 0.9, 0.001)
# rate_n_req = 0.45
# rate_f_req = 0.4

def compute_pwr_region_given_req(xi_req, rate_n_req, rate_f_req):
    w_n_min = rate_n_req
    w_n_list = np.arange(w_n_min, 0.999, 0.001)
    min_pow_oma = []
    for n in range(len(epsilon_req)):
        e_r = epsilon_req[n]
        min_power = 1
        for m in range(len(w_n_list)):
            w_n = w_n_list[m]
            const1 = reverse_europali_func.compute(rate_n_req / w_n)
            const3 = reverse_europali_func.compute(e_r)
            const2 = reverse_cifar_func.compute(rate_f_req/(1-w_n))
            const4 = reverse_cifar_func.compute(xi_req)
            if const1 == False or const2 == False:
                continue
            if const3 < 0:
                const3 = 0
            if const1 < 0:
                const1 = 0
            if const2 < 0:
                const2 = 0
            p_n_min = np.max([0, const1/ factor_n*w_n, const3/factor_n*w_n])
            # if p_n_min > 1:
            #     continue
            p_f_min = np.max([0, const2/factor_f*(1-w_n), const4/factor_f*(1-w_n)])
            # if p_f_min > 1:
            #     continue
            p_t_min = p_n_min + p_f_min
            # print(p_t_min)
            
            if p_t_min < min_power:
                min_power = p_t_min
                # print(w_n, p_n_min, 1-w_n, p_f_min)
        min_pow_oma.append(min_power)
    min_pow_noma = []
    # barpn = np.arange(0, 1, 0.001)
    # for n in range(len(xi_req)):
    #     xi_r = xi_req[n]
    #     min_noma = 1
    #     for n in range(len(barpn)):
    #         pn = barpn[n]
    #         pfmin = np.max([0, reverse_europali_func(xi_r)*(1/factor_f + pn), reverse_europali_func(rate_f_req)*(1/factor_f+pn)])
    #         pfmax = np.min([pn / reverse_cifar_func(rate_n_req)-1/factor_n, pn/reverse_cifar_func(epsilon_req)-1/factor_n, 1])
    #         if pfmin > pfmax or pn + pfmin > 1 or pfmax < 0 or pfmin > 1:
    #             continue
    #         if pfmin + pn < min_noma:
    #             min_noma = pfmin + pn
    #     min_pow_noma.append(min_noma)
    # plt.plot(xi_req, min_pow_noma, label = 'NOMA')
    # plt.plot(xi_req, min_pow_oma, label = 'OMA')
    # plt.xlabel(r"$Target\  \xi_{Cr}^{I,req}$")
    # plt.ylabel("Minimum Transmit Power")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # plt.savefig("power_region.png",dpi=300)
    for n in range(len(epsilon_req)):
        e_r = epsilon_req[n]
        pn_min_1 = reverse_europali_func.compute(rate_n_req) / factor_n
        pn_min_3 = reverse_europali_func.compute(e_r) / factor_n
        # if pn_min_1 > 1 or pn_min_1 < 0 or pn_min_3 > 1 or pn_min_3 < 0:
        #     print("Error")
        #     sys.exit(0)
        if pn_min_1 == False or pn_min_3 == False:
            print("Error, pn_min")
            sys.exit(0)
        if pn_min_1 < 0:
            pn_min_1 = 0
        if pn_min_3 < 0:
            pn_min_3 = 0
        pn_min = np.max([pn_min_1, pn_min_3])
        pn_range = np.arange(pn_min, 1, 0.001)
        min_noma_p = 1 
        for m in range(len(pn_range)):
            pn_bar = pn_range[m]
            pf_min_2 = reverse_cifar_func.compute(rate_f_req) * (1/factor_f+pn_bar)
            pf_min_4 = reverse_cifar_func.compute(xi_req) * (1/factor_f+pn_bar)
            if pf_min_2 == False or pf_min_4 == False:
                print("Error, pf_min")
                sys.exit(0)
            if pf_min_2 < 0:
                pf_min_2 = 0
            if pf_min_4 < 0:
                pf_min_4 = 0
            pf_min = np.max([pf_min_2, pf_min_4])
            sum_pow = pn_bar + pf_min
            if sum_pow < min_noma_p:
                min_noma_p = sum_pow
        min_pow_noma.append(min_noma_p)
    return min_pow_noma, min_pow_oma
# xi_req, rate_n_req, rate_f_req
min_pow_noma_low, min_pow_oma_low = compute_pwr_region_given_req(0.65, 0.5, 0.25)
min_pow_noma_med, min_pow_oma_med = compute_pwr_region_given_req(0.68, 0.55, 0.33)
min_pow_noma_high, min_pow_oma_high = compute_pwr_region_given_req(0.75, 0.6, 0.4)
plt.plot(epsilon_req, min_pow_noma_low, label = 'NOMA, low_req', c='g', linestyle='-.',linewidth=2)
plt.plot(epsilon_req, min_pow_oma_low, label = 'OMA, low_req', c='g', linestyle='--',linewidth=2)
plt.plot(epsilon_req, min_pow_noma_med, label = 'NOMA, med_req', c='b', linestyle='-.',linewidth=2)
plt.plot(epsilon_req, min_pow_oma_med, label = 'OMA, med_req', c='b', linestyle='--',linewidth=2)
plt.plot(epsilon_req, min_pow_noma_high, label = 'NOMA, high_req', c='r', linestyle='-.',linewidth=2)
plt.plot(epsilon_req, min_pow_oma_high, label = 'OMA, high_req', c='r', linestyle='--',linewidth=2)
# plt.xlabel(r"$Target\  \xi_{K}^{S,req}$", fontdict=font1)
# plt.ylabel("Minimum Transmit Power (MW)", fontdict=font1)
plt.xlabel(r"$Target\  \xi_{K}^{S,req}$")
plt.ylabel("Minimum Transmit Power (MW)")
plt.legend()
# plt.legend(bbox_to_anchor=(0.6, 0.1),  ncol=1)
plt.grid()
plt.show()
# plt.savefig("power_region.eps", format='eps', dpi=300, bbox_inches='tight')





# hn_gain = 1
# hf_gain = 0.5
# rou_snr = 20
# # cifar_func = logistic_increase_function(0.36606908, 0.90520419, 0.43258331, -1.14715606)
# # europali_func = logistic_increase_function(0.17723744, 0.95456146, 0.2981444, 1.68683409)
# rou_n = np.arange(0, 1, 0.001)
# noma_n = europali_func(rou_n * rou_snr * hn_gain)
# noma_f = cifar_func((1 - rou_n) * rou_snr * hf_gain /(rou_n * hf_gain + 1))
# oma_n = europali_func(hn_gain * rou_snr * rou_n)
# oma_n_max = europali_func(hn_gain * rou_snr)
# oma_f_max = cifar_func(hf_gain * rou_snr)
# oma_n_min = europali_func(0)
# oma_f_min = cifar_func(0)
# k = (oma_f_max - oma_f_min) / (oma_n_min - oma_n_max)
# oma_f = oma_f_min + (oma_n - oma_n_max) * k
# plt.plot(noma_n, noma_f, label = 'NOMA')
# plt.plot(oma_n, oma_f, label = 'OMA')
# plt.xlabel(r"$R_{N}, \times \frac{I_{N}}{KL}$")
# plt.ylabel(r"$R_{F}, \times \frac{I_{F}}{Pr}$")
# plt.legend()
# plt.grid()
# plt.show()


# #P|h1|^2/N0 = 1, P|h2|^2/N0 = 100
# Wmax = 1
# Pmax = 1
# factor1 = 1
# factor2 = 100
# r1max = np.log2(1+factor1)
# r1min = np.log2(1+0)
# r1 = []
# r2 = []
# r1req_l = np.arange(r1min, r1max, 0.001)
# W_l = np.arange(0, Wmax, 0.001)
# for n in range(len(r1req_l)):
#     r1.append(r1req_l[n])
#     r2max = -1
#     for m in range(len(W_l)):
#         alpha = W_l[m]
#         beta_min = np.minimum(alpha * (2**(r1req_l[n]/ alpha) - 1) / factor1 / Pmax, 1)
#         r2_c = (1 - alpha) * np.log2(1 + (1 - beta_min)*Pmax*factor2/(1-alpha))
#         if r2_c > r2max:
#             r2max = r2_c
#     r2.append(r2max)

# r1noma = []
# r2noma = []
# beta_l = np.arange(0, 1, 0.001)
# for n in range(len(beta_l)):
#     beta = beta_l[n]
#     r1_c = np.log2(1 +  beta / ((1-beta) + 1 / factor1))
#     r2_c = np.log2(1 + (1 - beta) * factor2)
#     r1noma.append(r1_c)
#     r2noma.append(r2_c)
# plt.figure()
# plt.plot(r1, r2, label = 'OMA')
# plt.plot(r1noma, r2noma, label = 'NOMA')
# plt.legend()
# plt.grid()
# plt.show()

