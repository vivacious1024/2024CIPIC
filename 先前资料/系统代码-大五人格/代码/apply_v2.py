import math
import os
import random

import numpy as np
import scipy.io
import torch

import FE


def seconds(time):
    """转为毫秒数"""
    return ((time[0] * 60 + time[1]) * 60 + time[2]) * 1000 + time[3]


def sub_time(begin, end):
    """计算时间差，时间都是数组形式，[时，分，秒，毫秒]，返回差值，单位：毫秒"""
    sub = seconds(end) - seconds(begin)
    return sub

def basenorm(eegarr):
    """基线校准，前0.2s"""
    channel = eegarr.shape[0]
    for i in range(channel):
        eegarr[i,:] = eegarr[i,:] - np.mean(eegarr[i, :100])
    return eegarr

def getpsd(eeg, eegTime, sampleRate):
    for i in range(2):
        delta, theta, alpha, beta, gamma = FE.DIV(eeg[i], eegTime, sampleRate)
        corpsd = np.vstack((FE.cor_X(delta), FE.cor_X(theta)))
        corpsd = np.vstack((corpsd, FE.cor_X(alpha)))
        corpsd = np.vstack((corpsd, FE.cor_X(beta)))
        corpsd = np.vstack((corpsd, FE.cor_X(gamma)))
        if i == 0:
            temp1 = corpsd.copy()
        else:
            temp2 = corpsd.copy()
    corpsd = np.vstack((temp1, temp2))

    return corpsd

def psdcom(r, time):
    '''组合功率谱密度特征'''
    for rroot, rdirs, rfiles in os.walk(r, topdown=False):
        if rroot != r:
            for root, dirs, files in os.walk(rroot, topdown=False):
                dirname = root.split('\\')[2]
                for file in files:
                    if '.mat' in file:
                        matpath = root + '\\' + file
                        beginTime = file.replace('.', '_').split('_')[1]
            data = scipy.io.loadmat(matpath)
            eegArr = np.array(data['eegdata'])
            sampleRate = 500

            '''划分epochs'''
            eegBegin = []
            for i in np.arange(4, 7) * 2:
                eegBegin.append(beginTime[i:i + 2])
            eegBegin.append(beginTime[14:17])
            eegBegin = list(map(int, eegBegin))  # eeg文件开始时间
            txtBegin = []
            for i in np.arange(0, 3) * 2:
                txtBegin.append(time[i:i + 2])
            txtBegin.append(time[6:9])
            txtBegin = list(map(int, txtBegin))  # eeg文件开始时间
            beforeTime = sub_time(eegBegin, txtBegin)  # 开始实验之前的时间

            eegnp = eegArr[:, math.ceil(beforeTime / 2.0):]  # 在实验范式之间采集到的信号
            eegnp = eegnp[:, 3000:]  # 提示语时间
            eegLength = eegnp.shape[1]
            eegTime = np.linspace(0, eegLength / float(sampleRate),
                                  num=eegLength)  # Time两数间隔0.002s，最后一位为eegLength*2/1000

            eegnp = FE.fda(eegnp, fpass1=2, fstop1=1, fpass2=50, fstop2=53, btype='bandpass', fs=500)

            psdarr = np.empty((3, 10, 610))
            for i in range(3):  # 按情绪词分成四类epoch
                tempeeg = (basenorm(eegnp[:, 610*i:610*(i+1)])+basenorm(eegnp[:, 610*(i+3):610*(i+4)])+basenorm(eegnp[:, 610*(i+6):610*(i+7)]))/3  # 一个trail的长度
                psdarr[i,:,:] = getpsd(tempeeg, eegTime[610*i:610*(i+1)], sampleRate)

    return psdarr


def apply(root, time, modelpath):
    psd = psdcom(root, time)
    model = torch.load(modelpath)
    data = torch.from_numpy(psd)
    model.eval()
    with torch.no_grad():
        predict = model(data)
        predict = (predict*40/100)+20
        print(predict)

    return predict


if __name__ == "__main__":
    root = ''  # 数据根目录
    time = 0  # 视频开始时间
    modelpath = ''
    result = apply(root, time, modelpath)


