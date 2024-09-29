import math
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import FE


def list_time(ls):
    """将txt文件记录的程序开始时间转为方便操作的数组"""
    time = ls[3].replace('.', ':').split(':')
    if ls[4] == 'PM':
        time[0] = str(int(time[0]) + 12)
    time = list(map(int, time))
    return time


def seconds(time):
    """转为毫秒数"""
    return ((time[0] * 60 + time[1]) * 60 + time[2]) * 1000 + time[3]


def sub_time(begin, end):
    """计算时间差，时间都是数组形式，[时，分，秒，毫秒]，返回差值，单位：毫秒"""
    sub = seconds(end) - seconds(begin)
    return sub


def group(r='F:\作业\本基项目\\val_data'):
    """将规定格式的原始数据集按被试和情绪词分段，每一个.mat文件都是一个epoch"""
    number = 0
    errcount = 0
    for rroot, rdirs, rfiles in os.walk(r, topdown=False):
        if rroot != r:
            for root, dirs, files in os.walk(rroot, topdown=False):
                dirname = root.split('\\')[4]
                for file in files:
                    if '.mat' in file:
                        matpath = root + '\\' + file
                        beginTime = file.replace('.', '_').split('_')[1]
                    elif '.txt' in file:
                        txtName = file
            '''读取eeg文件'''
            # matpath = 'F:\作业\本基项目\data\\2021071416\Src_20210714165808585.mat'
            data = scipy.io.loadmat(matpath)
            eegArr = np.array(data['eegdata'])
            sampleRate = 500

            '''读取txt记录数据'''
            file = open(root + '\\' + txtName, 'r')
            recordFile = []
            for line in file:
                splitFile = line.replace('\n', '')
                splitFile = splitFile.split('\t')
                recordFile.append(splitFile)
            recordFile[200] = "".join(recordFile[200]).replace(',', ' ')
            recordFile[200] = list(filter(None, recordFile[200].split(' ')))
            recordFile[201] = "".join(recordFile[201]).replace(',', ' ')
            recordFile[201] = list(filter(None, recordFile[201].split(' ')))  # 读取开始和结束的绝对时间

            '''划分epochs'''
            eegBegin = []
            for i in np.arange(4, 7) * 2:
                eegBegin.append(beginTime[i:i + 2])
            eegBegin.append(beginTime[14:17])
            eegBegin = list(map(int, eegBegin))  # eeg文件开始时间
            txtBegin = list_time(recordFile[200])  # txt文件开始时间
            txtEnd = list_time(recordFile[201])  # txt文件结束时间
            beforeTime = sub_time(eegBegin, txtBegin)  # 开始实验之前的时间
            expTime = sub_time(txtBegin, txtEnd)  # 实验范式花费的时间

            eegnp = eegArr[:, math.ceil(beforeTime / 2.0):math.ceil((beforeTime + expTime) / 2.0) + 1]  # 在实验范式之间采集到的信号

            eegnp = eegnp[:, 400:]  # 在第一个词出现之前有一秒的空白屏
            eegLength = eegnp.shape[1]
            eegTime = np.linspace(0, eegLength / float(sampleRate),
                                  num=eegLength)  # Time两数间隔0.002s，最后一位为eegLength*2/1000
            '''初步处理，傅里叶滤波'''
            # eegnp = FE.FFT(eegnp, [1, 50], eegTime)
            eegnp = FE.fda(eegnp, fpass1=2, fstop1=1, fpass2=50, fstop2=53, btype='bandpass', fs=500)

            groupCount = [0, 0, 0, 0]
            epochBegin = 0
            for i in range(200):  # 按情绪词分成四类epoch
                if not os.path.exists('F:\作业\本基项目\\val_group\\' + dirname + '\\' + recordFile[i][1]):
                    os.makedirs('F:\作业\本基项目\\val_group\\' + dirname + '\\' + recordFile[i][1])
                tempTime = float(recordFile[i][4]) + 1 + 0.020  # MatLab读入图片会有延迟，估算为0.02s
                epochBegin += int(1.2 * sampleRate + 10)
                if eegLength - epochBegin >= tempTime:
                    scipy.io.savemat(
                        'F:\作业\本基项目\\val_group\\' + dirname + '\\' + recordFile[i][1] + '\\' + str(
                            groupCount[int(recordFile[i][1]) - 1]) + '.mat',
                        {'epoch': eegnp[:, epochBegin:epochBegin + math.ceil(tempTime * sampleRate)],
                         'time': eegTime[epochBegin:epochBegin + math.ceil(tempTime * sampleRate)]})
                else:
                    print(epochBegin, '  ', eegLength)
                    break
                groupCount[int(recordFile[i][1]) - 1] += 1
                epochBegin = epochBegin + math.ceil(tempTime * sampleRate)
            number += 1
    return errcount


def screen(r='F:\作业\本基项目\\val_group'):
    """筛选"""
    ampth = 500
    epochCount = [0 for x in range(108)]
    number = 0  # 每个样本筛掉的trail数量
    item = 0  # 样本数
    for rroots in os.listdir(r):
        if rroots != r:
            for roots, dirs, files in os.walk(r + '\\' + rroots, topdown=False):
                for file in files:
                    data = scipy.io.loadmat(roots + '\\' + file)
                    eegArr = data['epoch']
                    time = data['time']
                    for i in range(eegArr.shape[0]):
                        eegArr[i] = eegArr[i] - np.mean(eegArr[i, 0:100])
                    wordclass = roots.split('\\')[-1]
                    if (wordclass != '4') and (np.max(np.abs(eegArr)) > ampth or eegArr.shape != (2, 610)):
                        number += 1
                        epochCount[item] += 1
                        os.remove(roots + '\\' + file)
                    else:
                        scipy.io.savemat(roots + '\\' + file, {'epoch': eegArr, 'time': time})
            item += 1
    print(sum(epochCount))
    return epochCount


if __name__ == "__main__":
    print(group())
    print(screen())
    # [0, 0, 5, 38, 0, 42, 1, 146, 56, 0, 40, 31, 3, 3, 0, 0, 5, 0, 0, 25, 3, 1, 0, 0, 1, 6, 1, 4, 0, 0, 4, 8, 12, 1, 0,
    # 0, 0, 0, 3, 0, 4, 1, 1, 2, 1, 4, 0, 1, 4, 3, 7, 0, 1, 0, 0, 2, 3, 2, 0, 2, 1, 1, 2, 0, 10, 0, 0, 0, 1, 1, 0, 0, 5,
    # 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
