import FE
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import openpyxl
from scipy import signal
from scipy.fftpack import fft, ifft

if __name__ == "__main__":
    '''提取.mat数据'''
    matpath = 'F:\作业\本基项目\data\\2021070619\Src_20210706193006.mat'
    sampleRate = 500
    data = scipy.io.loadmat(matpath)
    eegArr = np.array(data['eegdata'])  # print(data['__header__'],'\n',data['__version__'],'\n',data['__globals__'])
    eegLength = len(eegArr[0])
    eegTime = np.linspace(0, eegLength * 2 / 1000.0, num=eegLength)  # Time两数间隔0.002s，最后一位为eegLenth*2/1000
    '''归一化'''
    eegArr_T = FE.maxminnorm(eegArr.T)
    eegArr = eegArr_T.T

    '''傅里叶滤波'''
    eegFilter = FE.fda(eegArr, fpass1=8,fstop1=5, fpass2=50, fstop2=53, btype='bandpass', fs=500)
    '''去伪迹'''
    windowLen = 100
    eegOne = eegFilter[1, 4000:5200]
    timeOne = eegTime[4000:5200]
    plt.figure()
    plt.plot(eegTime[4000:5200], eegOne)
    plt.show()
    eegSSA = np.empty((7, len(eegOne)))
    y_low = eegOne
    for i in range(6):
        print("-----%d阶ssa------" % (i + 1))
        y_high, y_low = FE.ssa(y_low, sampleRate, windowLen, i + 1)
        eegSSA[i] = y_high
    eegSSA[i + 1] = y_low
    plt.figure(2)
    for i in range(7):
        plt.subplot(7, 1, i + 1)
        plt.plot(timeOne, eegSSA[i])
    plt.show()

    # ICA
    eegICA, W, White = FE.ICA(eegSSA)
    plt.figure(3)
    for i in range(7):
        plt.subplot(7, 1, i + 1)
        plt.plot(timeOne, eegICA[i])
    plt.show()

    # 根据样本熵判断伪迹成分
    sample_en = np.empty((eegSSA.shape[0]))
    print(sample_en.shape)
    for i in range(eegSSA.shape[0]):
        sample_en[i] = FE.sample_entropy(eegICA[i])
    sample_en_index = np.argsort(sample_en)
    sample_en = sample_en[sample_en_index]
    print(sample_en)
    judge = 0
    ica_remove = []
    for k in range(1, int(eegSSA.shape[0] / 2)):
        if sample_en[k + 1] - sample_en[k] < sample_en[k] - sample_en[k - 1]:
            ica_remove.append(sample_en_index[k])
            judge += 1
    if judge == 0:
        ica_remove.append(sample_en_index[0])
        judge = 1
    # 去除伪迹
    for i in ica_remove:
        eegICA[i, :] = 0

    # 重构ICA
    eegInvICA=FE.invICA(eegICA,W,White)
    eegActual=np.sum(eegICA,axis=0)
    for i in range(eegActual.shape[0]):
        eegActual[i]=(eegActual[i]-np.min(eegActual))/(np.max(eegActual)-np.min(eegActual))
    eegActual=FE.FFT(eegActual,[1,50],timeOne)
    plt.plot(timeOne,eegOne)
    plt.plot(timeOne,eegActual,'r')
    plt.show()

