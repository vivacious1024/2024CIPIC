import os
import random

import numpy as np
import scipy.io
from pandas.core.frame import DataFrame

import FE

random.seed(10)

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

def psdmat(path):
    """读取mat数据，提取每段数据的五种4波段的功率谱密度作为特征"""
    eegLength = 610
    sampleRate = 500
    eegArr = np.zeros((2, eegLength))
    eegTime = np.linspace(0, eegLength * 2 / 1000.0, num=eegLength)
    count = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if '.mat' in file:
                matpath = root + '\\' + file
                dirname = root.split('\\')
                data = scipy.io.loadmat(matpath)
                if data['epoch'].shape != (2, 610):
                    print(root)
                    print(file)
                    continue
                temp= np.array(data['epoch'])
                eegpsd = getpsd(temp, eegTime, sampleRate)
                if not os.path.exists('F:\作业\本基项目\\val_psd\\' + dirname[-2] + '\\' + dirname[-1]):
                    os.makedirs('F:\作业\本基项目\\val_psd\\' + dirname[-2] + '\\' + dirname[-1])
                scipy.io.savemat('F:\作业\本基项目\\val_psd\\' + dirname[-2] + '\\' + dirname[-1] + '\\' + file,
                                 {'epoch': eegpsd})
                count += 1


def psd_save(path='F:\作业\本基项目\\val_group'):
    """遍历所有.mat文件，映射到设计好的特征矩阵，并存储起来"""
    count = 0
    for rroot, rdirs, rfiles in os.walk(path, topdown=False):
        if len(rdirs) == 4:
            dirname = rroot.split('\\')[-1]
            for root, dirs, files in os.walk(rroot, topdown=False):
                if root != rroot:
                    filename = root.split('\\')[-1]
                    if filename != '4':
                        psdmat(root)
                        count += 1

    return count


def psd_com(path):
    """将三种psd各选一个进行组合(全组合)"""
    # target = 'F:\作业\本基项目\\psdcom\\'
    target = 'H:\psdcom2\\'
    for dirr in os.listdir(path):
        print('-------',dirr,'--------')
        source = path+'\\'+dirr+'\\'
        number = 0
        if not os.path.exists(target+dirr):
            os.makedirs(target+dirr)
        positives = os.listdir(source+'1')
        random.shuffle(positives)
        for positive in positives:
            data_pos = scipy.io.loadmat(source + '1\\'+positive)
            print('positive number:'+positive)
            neuters = os.listdir(source+'2')
            random.shuffle(neuters)
            for neuter in neuters:
                data_neu = scipy.io.loadmat(source + '2\\' + neuter)
                negatives = os.listdir(source+'3')
                random.shuffle(negatives)
                for negative in negatives:
                    data_neg = scipy.io.loadmat(source + '3\\' + negative)
                    com = np.vstack((data_pos, data_neu, data_neg))  # 组合
                    scipy.io.savemat(target+dirr+'\\'+str(number)+'.mat', {'epoch': com})
                    if number == 27000:
                        break
                    number += 1

def randompsd(path):
    kind = os.listdir(path)
    random.shuffle(kind)
    for sample in kind:
        data = scipy.io.loadmat(path+'\\'+sample)['epoch']
        yield data

def psd_com_part(path):
    """将三种psd各选一个进行组合(部分组合)"""
    target = 'F:\作业\本基项目\\val_psdcom\\'
    # target = 'H:\psdcom2\\'
    for dirr in os.listdir(path):
        print('-------',dirr,'--------')
        source = path+'\\'+dirr+'\\'
        number = 0
        if not os.path.exists(target+dirr):
            os.makedirs(target+dirr)
        positives = randompsd(source+'1')
        neuters = randompsd(source + '2')
        negatives = randompsd(source + '3')
        for data_pos, data_neu, data_neg in zip(positives, neuters, negatives):
            com = np.vstack((data_pos, data_neu, data_neg))  # 组合
            scipy.io.savemat(target + dirr + '\\' + str(number) + '.mat', {'epoch': com})
            number += 1


if __name__ == "__main__":
    """
    读取数据，对相同情绪词性的trail做平均
    按编号存储特征矩阵，2（channel）*3024（feature）
    """
    # psd_save()
    # psd_com('F:\作业\本基项目\\val_psd')
    psd_com_part('F:\作业\本基项目\\val_psd')