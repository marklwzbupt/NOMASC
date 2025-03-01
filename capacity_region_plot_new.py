import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


font1 = {'family': 'Times New Roman',
        'color':  'blue',
        'weight': 'normal',
        'size': 10,
        }
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
# 用户n是图像用户，用户f是文本用户]
# K = 16
# r = 0.33    
factor_n = 20
factor_f = 16
epsilon_req = 0.6
xi_req = 0.7
def give_result_under_K_and_Cr(K, r):
    cifar_func = Logistic_func(cifar_params[np.searchsorted(r_list, r)])
    reverse_cifar_func = Reverse_logistic_func(cifar_params[np.searchsorted(r_list, r)])
    europali_func = Logistic_func(europarl_params[int(np.log2(K)-1)])
    reverse_europali_func = Reverse_logistic_func(europarl_params[int(np.log2(K)-1)])

    # epsilon_req = 0.1
    # xi_req = 0.
    # rho_n_min = reverse_cifar_func.compute(epsilon_req) * (factor_n + 1) /(factor_n + 
    #             reverse_cifar_func.compute(epsilon_req) * factor_n)  
    if reverse_cifar_func.compute(xi_req) == False or reverse_europali_func.compute(epsilon_req) == False:
        print("Error, rho_n_min or rho_f_min is False!!!")
        sys.exit(0)
    rho_n_min = np.max([reverse_europali_func.compute(epsilon_req) / factor_n, 0])   
    if reverse_cifar_func.compute(xi_req) < 0:
        rho_f_min = 0
    else:
        rho_f_min = reverse_cifar_func.compute(xi_req) * (factor_f + 1) / (factor_f + 
                reverse_cifar_func.compute(xi_req) * factor_f)    
    if rho_n_min + rho_f_min >= 1:
        print("Error, rho_n_min + rho_f_min >= 1!!!")
        sys.exit(0)
    rho_f_max = np.min([1 - rho_n_min, 1])
    r_n = []
    r_f = []
    rho_f_list = np.arange(rho_f_min, rho_f_max, 0.001)
    for n in range(len(rho_f_list)):
        rho_f = rho_f_list[n]
        rho_n = 1 - rho_f
        r_f_c = cifar_func.compute(rho_f * factor_f / (rho_n * factor_f + 1)) / r
        r_n_c = europali_func.compute(rho_n * factor_n) / K
        r_n.append(r_n_c)
        r_f.append(r_f_c)

    r_n_max = europali_func.compute(factor_n) 
    r_f_max = cifar_func.compute(factor_f) 
    r_n_req_list = np.arange(0.001, r_n_max, 0.001)
    r_f_list = []
    for n in range(len(r_n_req_list)):
        r_n_req = r_n_req_list[n]
        w_n_min = r_n_req
        w_n_max = 1
        w_n_list = np.arange(w_n_min, w_n_max, 0.001)
        rf_max = 0
        for m in range(len(w_n_list)):
            w_n = w_n_list[m]
            const = reverse_europali_func.compute(r_n_req/w_n)
            if const == False:
                continue
            rho_n_min = np.max([0, const/factor_n*w_n, reverse_europali_func.compute(epsilon_req)/factor_n*w_n])
            # print(rho_n_min)
            if rho_n_min > 1:
                continue
            # print(rho_n_min)
            rho_n_up = np.min([reverse_europali_func.compute(epsilon_req) / factor_f * (1-w_n) + 1, 1])
            r_f_c = (1 - w_n) * cifar_func.compute((1 - rho_n_min)*factor_f / (1-w_n))
            if r_f_c > rf_max:
                rf_max = r_f_c
        r_f_list.append(rf_max / r)
    return r_n, r_f, r_f_list, r_n_req_list/K
r_n_1, r_f_1, r_f_list_1, r_n_req_list_1 = give_result_under_K_and_Cr(16, 0.08)

r_n_2, r_f_2, r_f_list_2, r_n_req_list_2 = give_result_under_K_and_Cr(16, 0.17)

r_n_3, r_f_3, r_f_list_3, r_n_req_list_3 = give_result_under_K_and_Cr(16, 0.25)



# plt.plot(r_f, r_n, label = 'NOMA')
# plt.plot(r_f_list, r_n_req_list / r, label = 'OMA')
plt.plot(r_f_1, r_n_1,c='g', linestyle='-.', label = 'NOMA, Cr=0.08')
plt.plot(r_f_list_1, r_n_req_list_1,c='g', linestyle='-',label = 'OMA, Cr=0.08')
plt.plot(r_f_2, r_n_2,c='r', linestyle='-.', label = 'NOMA, Cr=0.17')
plt.plot(r_f_list_2, r_n_req_list_2,c='r', linestyle='-',label = 'OMA, Cr=0.17')
plt.plot(r_f_3, r_n_3,c='b', linestyle='-.', label = 'NOMA, Cr=0.25')
plt.plot(r_f_list_3, r_n_req_list_3,c='b', linestyle='-', label = 'OMA, Cr=0.25')
# plt.xlabel(r"$\Gamma_{F}\ (\ Msuts/s\ )$", fontdict=font1)
# plt.ylabel(r"$\Gamma_{N}\ (\ Msuts/s\ )$", fontdict=font1)
plt.xlabel(r"$\Gamma_{F}\ (\ Msuts/s\ )$")
plt.ylabel(r"$\Gamma_{N}\ (\ Msuts/s\ )$")
# 画两条虚线
plt.plot([r_f_1[0], r_f_1[0]], [0, r_n_1[0]], c='k', linestyle='--')
plt.plot([r_f_1[-1], r_f_1[-1]], [0, r_n_1[-1]], c='k', linestyle='--')

plt.plot([r_f_2[0], r_f_2[0]], [0, r_n_2[0]], c='k', linestyle='--')
plt.plot([r_f_2[-1], r_f_2[-1]], [0, r_n_2[-1]], c='k', linestyle='--')

plt.plot([r_f_3[0], r_f_3[0]], [0, r_n_3[0]], c='k', linestyle='--')
plt.plot([r_f_3[-1], r_f_3[-1]], [0, r_n_3[-1]], c='k', linestyle='--')
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'+r"$\times\frac{I_{I}}{L_{I}}$"))

plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'+r"$\times\frac{I_{S}}{L_{S}}$"))

# # 座标轴调位
# ax = plt.gca()
# # 移到原点
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data', 0))
# plt.scatter([r_f[0]],[r_n[0]],s=25,c='r') 
# plt.text(r_f[0]+0.01, r_n[0]+0.01, r"$(\Gamma_{F}^{min}, \Gamma_{N}^{max})$", ha='center', va='bottom', fontsize=10.5)  # horizontal alignment

# plt.scatter([r_f[-1]],[r_n[-1]],s=25,c='r') 
# plt.text(r_f[-1], r_n[-1]-0.1, r"$(\Gamma_{F}^{max}, \Gamma_{N}^{min})$", ha='center', va='bottom', fontsize=10.5)  # horizontal alignment
# t=plt.text(0.11, -1,r'$\times \frac{I_{I}}{L_{I}}$',fontsize=10, horizontalalignment='center',verticalalignment='center')
# #在这里设置是text的旋转，0为水平，90为竖直
# t.set_rotation(0)
# t=plt.text(-0.0105, 10.9,r'$\times \frac{I_{S}}{L_{S}}$',fontsize=10, horizontalalignment='center',verticalalignment='center')
#在这里设置是text的旋转，0为水平，90为竖直
# t.set_rotation(90)
plt.legend(loc = 3)
plt.grid()
# plt.show()
plt.savefig("capacity_region.eps", format='eps', dpi = 300, bbox_inches='tight')









# hn_gain = 1
# hf_gain = 0.1
# rou_snr = 20
# # cifar_func = logistic_increase_function(0.36606908, 0.90520419, 0.43258331, -1.14715606)
# # europali_func = logistic_increase_function(0.17723744, 0.95456146, 0.2981444, 1.68683409)
# rou_n = np.arange(0, 1, 0.001)
# noma_n = cifar_func(rou_n * rou_snr * hn_gain)
# noma_f = europali_func((1 - rou_n) * rou_snr * hf_gain /(rou_n * hf_gain * rou_snr + 1))
# oma_n = cifar_func(hn_gain * rou_snr * rou_n)
# oma_n_max = cifar_func(hn_gain * rou_snr)
# oma_f_max = europali_func(hf_gain * rou_snr)
# oma_n_min = cifar_func(0)
# oma_f_min = europali_func(0)
# k = (oma_f_max - oma_f_min) / (oma_n_min - oma_n_max)
# oma_f = oma_f_min + (oma_n - oma_n_max) * k
# plt.plot(noma_n, noma_f, label = 'NOMA')
# plt.plot(oma_n, oma_f, label = 'OMA')
# plt.xlabel(r"$R_{N}, \times \frac{I_{N}}{KL}$")
# plt.ylabel(r"$R_{F}, \times \frac{I_{F}}{Pr}$")
# plt.legend()
# plt.grid()
# plt.show()


# # #P|h1|^2/N0 = 1, P|h2|^2/N0 = 100
# # Wmax = 1
# # Pmax = 1
# # factor1 = 1
# # factor2 = 100
# # r1max = np.log2(1+factor1)
# # r1min = np.log2(1+0)
# # r1 = []
# # r2 = []
# # r1req_l = np.arange(r1min, r1max, 0.001)
# # W_l = np.arange(0, Wmax, 0.001)
# # for n in range(len(r1req_l)):
# #     r1.append(r1req_l[n])
# #     r2max = -1
# #     for m in range(len(W_l)):
# #         alpha = W_l[m]
# #         beta_min = np.minimum(alpha * (2**(r1req_l[n]/ alpha) - 1) / factor1 / Pmax, 1)
# #         r2_c = (1 - alpha) * np.log2(1 + (1 - beta_min)*Pmax*factor2/(1-alpha))
# #         if r2_c > r2max:
# #             r2max = r2_c
# #     r2.append(r2max)

# # r1noma = []
# # r2noma = []
# # beta_l = np.arange(0, 1, 0.001)
# # for n in range(len(beta_l)):
# #     beta = beta_l[n]
# #     r1_c = np.log2(1 +  beta / ((1-beta) + 1 / factor1))
# #     r2_c = np.log2(1 + (1 - beta) * factor2)
# #     r1noma.append(r1_c)
# #     r2noma.append(r2_c)
# # plt.figure()
# # plt.plot(r1, r2, label = 'OMA')
# # plt.plot(r1noma, r2noma, label = 'NOMA')
# # plt.legend()
# # plt.grid()
# # plt.show()

