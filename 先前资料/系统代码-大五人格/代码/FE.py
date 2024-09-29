# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:18:48 2020

@author: YangSh

@code_describe : feature_extract function


"""
from __future__ import division
import numpy as np

np.set_printoptions(threshold=np.inf)
import pywt
import math
import nolds
from pyentrp import entropy as ent
import scipy
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.fftpack import fft, ifft, fftfreq, fftshift
from scipy import signal
from PyEMD import EMD, Visualisation
from numpy import linalg as LA
import sys
import pylab as plt

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# time domain
def min_X(X):
    return np.min(X)


def max_X(X):
    return np.max(X)


def std_X(X):
    return np.std(X)


def mean_X(X):
    return np.mean(X)


def var_X(X):
    return np.var(X)


'''二维数组的归一化
    每行一个样本，每列为一个特征
'''


def maxminnorm(X):
    maxcols = X.max(axis=0)
    mincols = X.min(axis=0)
    data_shape = X.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (X[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    t = np.nan_to_num(t)
    return t


# 总变分，用于平滑信号，消除一些噪声
def totalVariation(X):
    Max = np.max(X)
    Min = np.min(X)
    return np.sum(np.abs(np.diff(X))) / ((Max - Min) * (len(X) - 1))


# 偏度用来度量分布是否对称。正态分布左右是对称的，偏度系数为0。较大的正值表明该分布具有右侧较长尾部。较大的负值表明有左侧较长尾部
def skew_X(X):
    skewness = skew(X)
    return skewness


# 峰度系数（Kurtosis）用来度量数据在中心聚集程度。在正态分布情况下，峰度系数值是3。
# >3的峰度系数说明观察量更集中，有比正态分布更短的尾部；
# <3的峰度系数说明观测量不那么集中，有比正态分布更长的尾部，类似于矩形的均匀分布。
def kurs_X(X):
    kurs = kurtosis(X)
    return kurs


# 常用均方根值来分析噪声
def rms_X(X):
    RMS = np.sqrt((np.sum(np.square(X))) * 1.0 / len(X))
    return RMS


def peak_X(X):
    Peak = np.max([np.abs(max_X(X)), np.abs(min_X(X))])
    return Peak


# 峰均比
def papr_X(X):
    Peak = peak_X(X)
    RMS = rms_X(X)
    PAPR = np.square(Peak) * 1.0 / np.square(RMS)
    return PAPR


'''
计算时域10个bin特征
'''


# 过滤X中相等的点
def filter_X(X):
    X_new = []
    length = np.shape(X)[0]
    for i in range(1, length):
        if i != 0 and X[i] == X[i - 1]:
            continue
        X_new.append(X[i])
    return X_new


# 求X中所有的极大值和极小值点
def minmax_cal(X):
    length = np.shape(X)[0]
    min_value = []
    min_index = []
    max_value = []
    max_index = []
    first = ''
    for i in range(1, length - 1):
        if X[i] < X[i - 1] and X[i] < X[i + 1]:
            min_value.append(X[i])
            min_index.append(i)
        if X[i] > X[i - 1] and X[i] > X[i + 1]:
            max_value.append(X[i])
            max_index.append(i)
    if len(min_index) and len(max_index):
        if max_index[0] > min_index[0]:
            first = 'min'
        else:
            first = 'max'
        return min_value, max_value, first
    else:
        return None, None, None


# 计算所有的极大值和极小值的差值
def minmax_sub_cal(X):
    min_value, max_value, first = minmax_cal(X)
    if min_value and max_value and first:
        max_length = np.shape(max_value)[0]
        sub = []
        if first == 'min':
            for i in range(max_length - 1):
                sub.append(max_value[i] - min_value[i])
                sub.append(max_value[i] - min_value[i + 1])
        else:
            for i in range(1, max_length - 1):
                sub.append(max_value[i] - min_value[i - 1])
                sub.append(max_value[i] - min_value[i])
        return sub
    else:
        return None


# 计算极大极小值差值占比
def minmax_percent_cal(X, step=10):
    X = filter_X(X)
    sub = minmax_sub_cal(X)
    if sub:
        length = int(np.shape(sub)[0])
        max_value = max(sub)
        min_value = min(sub)
        diff = max_value - min_value
        value = diff / step
        nums = []
        sub = np.array(sub)
        for i in range(step):
            scale_min = sub >= min_value + i * value
            scale_max = sub < min_value + (i + 1) * value
            scale = scale_min & scale_max
            num = np.where(scale)[0]
            size = np.shape(num)[0]
            nums.append(size)
        nums[-1] = nums[-1] + sum(sub == max_value)
        nums = np.array(nums, dtype=int)
        per = nums / length
        return per
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# 非线性特征提取
# 采样频率为256Hz,则信号的最大频率为128Hz，进行5层小波分解
# 采样频率为500Hz,则信号的最大频率为250Hz，进行6层小波分解
def relativePower(X):
    Ca6, Cd6, Cd5, Cd4, Cd3, Cd2, Cd1 = pywt.wavedec(X, wavelet='db5', level=6)
    EA6 = sum([i * i for i in Ca6])
    ED6 = sum([i * i for i in Cd6])
    ED5 = sum([i * i for i in Cd5])
    ED4 = sum([i * i for i in Cd4])
    ED3 = sum([i * i for i in Cd3])
    ED2 = sum([i * i for i in Cd2])
    ED1 = sum([i * i for i in Cd1])
    E = EA6 + ED6 + ED5 + ED4 + ED3 + ED2 + ED1
    pEA6 = EA6 / E
    pED6 = ED6 / E
    pED5 = ED5 / E
    pED4 = ED4 / E
    pED3 = ED3 / E
    pED2 = ED2 / E
    pED1 = ED1 / E
    return pEA6, pED6, pED5, pED4, pED3, pED2, pED1


# nonlinear analysis
# 小波熵
def wavelet_entopy(X):
    [pEA6, pED6, pED5, pED4, pED3, pED2, pED1] = relativePower(X)
    wavelet_entopy = - (pEA6 * math.log(pEA6) + pED6 * math.log(pED6) + pED5 * math.log(pED5)
                        + pED4 * math.log(pED4) + pED3 * math.log(pED3) + pED2 * math.log(pED2) + pED1 * math.log(pED1))
    return wavelet_entopy


# 计算Detrended Fluctuation Analysis值
def DFA(X):
    y = nolds.dfa(X)
    return y


# 计算赫斯特指数
def Hurst(X):
    y = nolds.hurst_rs(X)
    return y


# 计算Petrosian's Fractal Dimension分形维数值
def Petrosian_FD(X):
    D = np.diff(X)

    delta = 0;
    N = len(X)
    # number of sign changes in signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            delta += 1

    feature = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * delta)))

    return feature


# 计算样本熵
def sample_entropy(X):
    y = nolds.sampen(X)
    return y


# 计算排列熵
# 度量时间序列复杂性的一种方法,排列熵H的大小表征时间序列的随机程度，值越小说明该时间序列越规则，反之，该时间序列越具有随机性。
def permutation_entropy(X):
    y = ent.permutation_entropy(X, 4, 1)
    return y


# Hjorth Parameter: mobility and complexity
def Hjorth(X):
    D = np.diff(X)
    D = list(D)
    D.insert(0, X[0])
    VarX = np.var(X)
    VarD = np.var(D)
    Mobility = np.sqrt(VarD / VarX)

    DD = np.diff(D)
    VarDD = np.var(DD)
    Complexity = np.sqrt(VarDD / VarD) / Mobility

    return Mobility, Complexity


def FFT(signal, low_high, time):
    if signal.shape[0] > 64:
        W = fftfreq(signal.size, d=(time[1] - time[0]))  # d*n==1
        f_signal = fft(signal)

        # If our original signal time was in seconds, this is now in Hz
        cut_f_signal = f_signal.copy()
        cut_f_signal[W > low_high[1]] = 0
        cut_f_signal[W < low_high[0]] = 0

        cut_signal = ifft(cut_f_signal)
        cut_signal = np.real(cut_signal)

    else:
        length = signal.shape[0]
        cut_signal = np.empty(signal.shape)
        for i in range(length):
            cut_signal[i, :] = FFT(signal[i], low_high, time)

    # plt.subplot(221)
    # plt.plot(time, signal)
    # plt.subplot(222)
    # plt.plot(W, abs(f_signal))
    # plt.xlim(0, 80)
    # plt.subplot(223)
    # plt.plot(W, abs(cut_f_signal))
    # plt.xlim(0, 80)
    # plt.subplot(224)
    # plt.plot(time, cut_signal)
    # plt.show()
    return cut_signal


# N阶的FIR数字滤波
def fda(x, fpass1, fstop1, fpass2,fstop2, btype, fs):  # （输入的信号，截止频率下限，截止频率上限）
    '''lowpass 低通， bandpass 带通, highpass 高通'''
    ws1 = (fpass1 + fstop1) / 2 / (fs / 2)
    ws2 = (fpass2 + fstop2) / 2 / (fs / 2)
    N = math.ceil(8 * math.pi / ((fstop2 - fpass2) / fs * 2 * math.pi))
    hn=scipy.signal.firwin(N,[ws1,ws2],window='hamming',pass_zero=btype)
    # plt.figure(1)
    # plt.plot(np.arange(0, math.ceil(N / 2)) * 2 / N, 20*np.log10(abs(np.fft.fft(hn, N))[0:math.ceil(N / 2)]))
    # plt.xlabel('频率/pi')
    # plt.ylabel('dB')
    # plt.grid()
    filtedData = scipy.signal.filtfilt(hn, 1, x)
    # plt.figure(2)
    # plt.plot([i/fs for i in range(x.shape[1])],x[0,:],'b',label='original')
    # plt.grid()
    # plt.plot([i/fs for i in range(x.shape[1])],filtedData[0,:],'r',label='filtered')
    # plt.legend(loc="upper right")
    # plt.xlabel('时间/s')
    # plt.ylabel('电压/uv')
    # filtedDatafft=20 * np.log10(abs(np.fft.fft(filtedData,filtedData.shape[1]))[0,0:math.ceil(filtedData.shape[1] / 2)])
    # plt.figure(3)
    # plt.plot(np.arange(0, math.ceil(filtedData.shape[1] / 2)) * 2 / filtedData.shape[1],filtedDatafft)
    # plt.grid()
    # plt.xlabel('频率/pi')
    # plt.ylabel('dB')
    # plt.show()
    return filtedData


# 自相关方法求功率谱密度
def cor_X(X):
    num_fft = len(X)
    cor_x = np.correlate(X, X, 'same')
    cor_x = fft(cor_x, num_fft)
    ps_cor = np.abs(cor_x)
    ps_cor = ps_cor / np.max(ps_cor)

    return ps_cor


# 求平均功率
def ave_power(X):
    return (1 / 2 * len(X)) * np.sum(np.power(X, 2))


def MyEMD(eeg_signal, time):
    #  提取imfs和剩余
    emd = EMD()
    emd.emd(eeg_signal)
    imfs, res = emd.get_imfs_and_residue()
    # 绘制 IMF
    # vis = Visualisation()
    # vis.plot_imfs(imfs=imfs, residue=res, t=time, include_residue=True)
    # # 绘制并显示所有提供的IMF的瞬时频率
    # vis.plot_instant_freq(time, imfs=imfs)
    # vis.show()

    return imfs, res


# 重构EDM
def invEDM(imfs_res):
    signal = np.sum(imfs_res, axis=0)

    # x = np.arange(0, imfs_res.shape[1])
    # ax1 = plt.subplot(421)
    # ax2 = plt.subplot(422)
    # ax3 = plt.subplot(423)
    # ax4 = plt.subplot(424)
    # ax5 = plt.subplot(425)
    # ax6 = plt.subplot(426)
    # ax7 = plt.subplot(427)
    # ax8 = plt.subplot(428)
    # ax1.plot(x, imfs_res.T[:, 0])
    # ax2.plot(x, imfs_res.T[:, 1])
    # ax3.plot(x, imfs_res.T[:, 2])
    # ax4.plot(x, imfs_res.T[:, 3])
    # ax5.plot(x, imfs_res.T[:, 4])
    # ax6.plot(x, imfs_res.T[:, 5])
    # ax7.plot(x, imfs_res.T[:, 6])
    # ax8.plot(x, imfs_res.T[:, 7])
    # plt.show()

    return signal


def ICA(mix):
    Maxcount = 100000  # %最大迭代次数
    Critical = 0.00001  # %判断是否收敛
    R, C = mix.shape

    average = np.mean(mix, axis=1)  # 计算行均值，axis=0，计算每一列的均值

    for i in range(R):
        mix[i, :] = mix[i, :] - average[i]  # 数据标准化，均值为零
    Cx = np.cov(mix)
    value, eigvector = np.linalg.eig(Cx)  # 计算协方差阵的特征值
    for i in range(value.shape[0]):
        if value[i] == 0:
            value[i] += sys.float_info.min  # 判断协方差矩阵的特征值是否含零
    # print(value**(-1/2))
    val = value ** (-1 / 2) * np.eye(R, dtype=float)
    White = np.dot(val, eigvector.T)  # 白化矩阵

    Z = np.dot(White, mix)  # 混合矩阵的主成分Z，Z为正交阵

    W = np.random.random((R, R))  # 4x4
    # W = 0.5 * np.ones([R, R])  # 初始化权重矩阵

    for n in range(R):
        count = 0
        WP = W[:, n].reshape(R, 1)  # 初始化
        LastWP = np.zeros(R).reshape(R, 1)  # 列向量;LastWP=zeros(m,1);
        while LA.norm(WP - LastWP, 1) > Critical:
            # print(count," loop :",LA.norm(WP-LastWP,1))
            count = count + 1
            LastWP = np.copy(WP)  # %上次迭代的值
            gx = np.tanh(LastWP.T.dot(Z))  # 行向量

            for i in range(R):
                tm1 = np.mean(Z[i, :] * gx)
                tm2 = np.mean(1 - gx ** 2) * LastWP[i]  # 收敛快
                # tm2=np.mean(gx)*LastWP[i]     #收敛慢
                WP[i] = tm1 - tm2
            # print(" wp :", WP.T )
            WPP = np.zeros(R)  # 一维0向量
            for j in range(n):
                WPP = WPP + WP.T.dot(W[:, j]) * W[:, j]
            WP.shape = 1, R
            WP = WP - WPP
            WP.shape = R, 1
            WP = WP / (LA.norm(WP))
            if (count == Maxcount):
                print("reach Maxcount，exit loop", LA.norm(WP - LastWP, 1))
                break
        print("loop count:", count)
        W[:, n] = WP.reshape(R, )
    SZ = W.T.dot(Z)

    # plot extract signal
    # x = np.arange(0, C)
    # ax1 = plt.subplot(421)
    # ax2 = plt.subplot(422)
    # ax3 = plt.subplot(423)
    # ax4 = plt.subplot(424)
    # ax5 = plt.subplot(425)
    # ax6 = plt.subplot(426)
    # ax7 = plt.subplot(427)
    # ax8 = plt.subplot(428)
    # ax1.plot(x, SZ.T[:, 0])
    # ax2.plot(x, SZ.T[:, 1])
    # ax3.plot(x, SZ.T[:, 2])
    # ax4.plot(x, SZ.T[:, 3])
    # ax5.plot(x, SZ.T[:, 4])
    # ax6.plot(x, SZ.T[:, 5])
    # ax7.plot(x, SZ.T[:, 6])
    # ax8.plot(x, SZ.T[:, 7])
    # plt.show()

    return SZ, W, White


# 重构FastICA
def invICA(SZ, W, White):
    # print(SZ)
    Z_new = np.linalg.pinv(W.T).dot(SZ)
    # print(Z_new)
    mix_new = np.linalg.pinv(White).dot(Z_new)

    return mix_new


'''各频段时域信号α（8－14Hz）、β（14－32Hz）、δ（1－4Hz）、θ（4－8Hz）、γ（32-50HZ）'''


def DIV(X, time, sample_rate):
    delta = FFT(X, [1, 4], time)
    theta = FFT(X, [4, 8], time)
    alpha = FFT(X, [8, 14], time)
    beta = FFT(X, [14, 32], time)
    gamma = FFT(X, [32, 50], time)

    return delta, theta, alpha, beta, gamma


'''短时傅里叶变换后求功率谱'''


def STFT_PSD(X, fs, nperseg=None):
    # 计算并绘制STFT的大小
    amp = np.max(np.abs(np.fft.fft(X)))
    f, t, Zxx = signal.stft(X, fs, nperseg=nperseg)
    c = plt.pcolormesh(t, f, np.abs(Zxx), vmin=-amp, vmax=amp, cmap='Blues_r', shading='auto')
    plt.colorbar(c)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    print(f.shape, t.shape, Zxx.shape)
    print(t)
    # 功率
    ps = np.empty([t.shape[0], f.shape[0]])
    for i in range(t.shape[0]):
        ps[i] = np.abs(Zxx.T[i]) ** 2 / nperseg
    ps = ps.T
    ps = maxminnorm(ps)
    d = plt.pcolormesh(t, f, np.abs(ps), vmin=0, vmax=1, cmap='Blues', shading='auto')
    plt.colorbar(d)
    plt.title('STFT PSD')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    return ps


def xcorr(data):
    """自定义自相关函数"""
    length = len(data)
    R = []
    for m in range(0, length):
        sum = 0.0
        for n in range(0, length - m):
            sum = sum + data[n] * data[n + m]
        R.append(sum / length)
    return R


def findpeak(arr):
    """返回峰值索引，幅值降序"""
    length = len(arr)
    min = np.min(arr)
    peaks = np.empty(length)
    for i in range(length):
        if i == 0:
            if arr[i] > arr[i + 1]:
                peaks[i] = arr[i]
            else:
                peaks[i] = min
        elif i == length - 1:
            if arr[i] > arr[i - 1]:
                peaks[i] = arr[i]
            else:
                peaks[i] = min
        else:
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                peaks[i] = arr[i]
            else:
                peaks[i] = min
    return np.argsort(-peaks)


def myburg(yn, sample_rate, hf=0):
    """Burg参数法功率谱估计"""
    N = len(yn)
    p = int(N / 2)

    # 求自相关
    R = xcorr(yn)

    # 初始条件
    rou = np.zeros(p + 1)
    rou[0] = R[0]
    k = np.zeros(p + 1)
    a = np.zeros([p + 1, p + 1])
    ef = np.zeros([p + 1, N])
    eb = np.zeros([p + 1, N])
    for n in range(0, N):
        ef[0][n] = yn[n]
        eb[0][n] = yn[n]

    # burg 递推
    for m in range(1, p + 1):
        shang = 0.0
        xia = 0.0
        for n in range(m, N):
            shang = shang + ef[m - 1][n] * eb[m - 1][n - 1]
            xia = xia + (ef[m - 1][n]) ** 2 + (eb[m - 1][n - 1]) ** 2
        k[m] = -2 * shang / xia
        a[m][m] = k[m]
        for n in range(m, N):
            ef[m][n] = ef[m - 1][n] + k[m] * eb[m - 1][n - 1]
            eb[m][n] = eb[m - 1][n - 1] + k[m] * ef[m - 1][n]
        if k[m] > 1:
            break
        if m > 1:
            for i in range(1, m):
                a[m][i] = a[m - 1][i] + k[m] * a[m - 1][m - i]
        rou[m] = rou[m - 1] * (1 - (k[m] ** 2))

    # 得到p+1个参数，求频率响应
    G2 = rou[p]
    ap = []
    for k in range(1, p + 1):
        ap.append(a[p][k])
    apk = np.insert(ap, 0, [1])
    G = np.sqrt(G2)

    # 计算频率响应
    w, h = scipy.signal.freqz(G, apk, worN=N)
    Hf = abs(h)
    Sx = Hf ** 2
    f = (w / (2 * np.pi)) * sample_rate

    # freq = fftfreq(N, d=1/sample_rate)
    # plt.subplot(311)
    # plt.plot(np.arange(0,N/sample_rate,step=1/sample_rate), R)
    # plt.subplot(312)
    # plt.plot(f, Sx)
    # plt.title("power spectrum")
    # plt.xlabel("f")
    # plt.ylabel("Sx")
    # plt.show()

    f_estimate = findpeak(Sx)[0] * (sample_rate / 2 / N)  # 求最大的主频
    # print("f 的估计值:f=", f_estimate)

    ##自相关谱估计
    # num_fft = N
    # cor_x = np.correlate(yn, yn, 'same')
    # cor_x = fft(cor_x, num_fft)
    # ps_cor = np.abs(cor_x)
    # ps_cor = ps_cor / np.max(ps_cor)
    # ax = plt.subplot(313)
    # ax.set_title('indirect method')
    # plt.plot(freq[0:int(N / 2)], 20 * np.log10(ps_cor[:num_fft//2]))
    # plt.show()

    if hf == 0:
        return f_estimate
    else:
        return f_estimate, Sx, f


def ssa(series, sample_rate, windowLen, order):
    '''SSA 奇异谱分析
        series为信号序列 sample_rate为采样频率
        windowLen为嵌入窗口长度
        order为阶次
    '''
    series = series - np.mean(series)  # 中心化(非必须)
    fs_th = sample_rate / float(pow(2, order + 1))  # 频率阈值

    # step1 嵌入
    # windowLen = 200  # 嵌入窗口长度
    seriesLen = len(series)  # 序列长度
    K = seriesLen - windowLen + 1
    X = np.zeros((windowLen, K))
    for i in range(K):
        X[:, i] = series[i:i + windowLen]

    # step2: svd分解， U和sigma已经按升序排序
    U, sigma, VT = np.linalg.svd(X, full_matrices=False)

    # 分组 低频和高频
    U_low = []
    U_high = []
    for i in range(U.shape[1]):
        mainfre = myburg(U[:, i], sample_rate)
        if mainfre <= fs_th:
            U_low.append(i)
        else:
            U_high.append(i)
        # print(mainfre)

    for i in range(VT.shape[0]):
        VT[i, :] *= sigma[i]
    A = VT

    # 重组
    rec = np.zeros((windowLen, seriesLen))
    for i in range(windowLen):
        for j in range(windowLen - 1):
            for m in range(j + 1):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (j + 1)
        for j in range(windowLen - 1, seriesLen - windowLen + 1):
            for m in range(windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= windowLen
        for j in range(seriesLen - windowLen + 1, seriesLen):
            for m in range(j - seriesLen + windowLen, windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (seriesLen - j)

    rrr = np.sum(rec[0:5, :], axis=0)  # 选择重构的部分，这里选了前5个
    r_low = np.sum(rec[U_low, :], axis=0)
    r_high = np.sum(rec[U_high, :], axis=0)
    # plt.figure()
    # x = np.arange(seriesLen) / sample_rate
    # for i in range(12):
    #     ax = plt.subplot(6, 2, i + 1)
    #     ax.plot(x, rec[i, :])

    # plt.figure(2)
    # plt.subplot(221)
    # plt.plot(x, series)
    # plt.subplot(222)
    # plt.plot(x, rrr)
    # plt.subplot(223)
    # plt.plot(x, r_low)
    # plt.subplot(224)
    # plt.plot(x, r_high)
    # plt.show()

    return r_low, r_high
