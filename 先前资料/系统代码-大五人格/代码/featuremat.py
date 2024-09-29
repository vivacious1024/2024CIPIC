import os

import numpy as np
import scipy.io
from pandas.core.frame import DataFrame

import FE


def getmat(path):
    """读取mat数据，将每段数据平均"""
    eegLength = 610
    sampleRate = 500
    eegArr = np.zeros((2, eegLength))
    eegTime = np.linspace(0, eegLength * 2 / 1000.0, num=eegLength)
    count = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if '.mat' in file:
                matpath = root + '\\' + file
                data = scipy.io.loadmat(matpath)
                if data['epoch'].shape != (2, 610):
                    print(root)
                    print(file)
                    continue
                eegArr += np.array(data['epoch'])
                count += 1

    eegArr = eegArr / count  # 平均trail
    """提取特征"""
    # 归一化
    eegArr = FE.maxminnorm(eegArr.T).T - 0.5
    # 均值
    mean = np.mean(eegArr, 1)
    # 标准差
    std = np.std(eegArr, 1)
    # 均方根值
    rms = np.empty((2, 1))
    rms[0] = FE.rms_X(eegArr[0])
    rms[1] = FE.rms_X(eegArr[1])
    # 偏度
    skew = np.empty((2, 1))
    skew[0] = FE.skew_X(eegArr[0])
    skew[1] = FE.skew_X(eegArr[1])
    # 峰度
    kurs = np.empty((2, 1))
    kurs[0] = FE.kurs_X(eegArr[0])
    kurs[1] = FE.kurs_X(eegArr[1])
    # 峰均比
    papr = np.empty((2, 1))
    papr[0] = FE.papr_X(eegArr[0])
    papr[1] = FE.papr_X(eegArr[1])
    # 计算极大极小值差值占比
    mmpc = np.empty((2, 10))
    mmpc[0] = FE.minmax_percent_cal(eegArr[0], 10)
    mmpc[1] = FE.minmax_percent_cal(eegArr[1], 10)
    # 不同波段的功率谱密度
    '''各频段时域信号α（8－14Hz）、β（14－32Hz）、δ（1－4Hz）、θ（4－8Hz）、γ（32-50HZ）'''
    for i in range(2):
        delta, theta, alpha, beta, gamma = FE.DIV(eegArr[i], eegTime, sampleRate)
        corpsd = np.hstack((FE.cor_X(delta), FE.cor_X(theta)))
        corpsd = np.hstack((corpsd, FE.cor_X(alpha)))
        corpsd = np.hstack((corpsd, FE.cor_X(beta)))
        corpsd = np.hstack((corpsd, FE.cor_X(gamma)))
        if i == 0:
            temp1 = corpsd.copy()
        else:
            temp2 = corpsd.copy()
    corpsd = np.vstack((temp1, temp2))
    # for i in range(2):
    #     delta, theta, alpha, beta, gamma = FE.DIV(eegArr[i], eegTime, sampleRate)
    #     corpsd = np.hstack((FE.ave_power(delta), FE.ave_power(theta)))
    #     corpsd = np.hstack((corpsd, FE.ave_power(alpha)))
    #     corpsd = np.hstack((corpsd, FE.ave_power(beta)))
    #     corpsd = np.hstack((corpsd, FE.ave_power(gamma)))
    #     if i == 0:
    #         temp1 = corpsd.copy()
    #     else:
    #         temp2 = corpsd.copy()
    # corpsd = np.vstack((temp1, temp2))

    # 小波熵
    wavent = np.empty((2, 1))
    wavent[0] = FE.wavelet_entopy(eegArr[0])
    wavent[1] = FE.wavelet_entopy(eegArr[1])
    # 去趋势化分析
    dfa = np.empty((2, 1))
    dfa[0] = FE.DFA(eegArr[0])
    dfa[1] = FE.DFA(eegArr[1])
    # 赫斯特指数
    hurst = np.empty((2, 1))
    hurst[0] = FE.Hurst(eegArr[0])
    hurst[1] = FE.Hurst(eegArr[1])
    # 分形维数
    petfd = np.empty((2, 1))
    petfd[0] = FE.Petrosian_FD((eegArr[0]))
    petfd[1] = FE.Petrosian_FD(eegArr[1])
    # 样本熵
    sament = np.empty((2, 1))
    sament[0] = FE.sample_entropy(eegArr[0])
    sament[1] = FE.sample_entropy(eegArr[1])
    # 排列熵
    perent = np.empty((2, 1))
    perent[0] = FE.permutation_entropy(eegArr[0])
    perent[1] = FE.permutation_entropy(eegArr[1])
    # Hjorth参数
    hjorth = np.empty((2, 2))
    hjorth[0] = FE.Hjorth(eegArr[0])
    hjorth[1] = FE.Hjorth(eegArr[1])

    """构建特征矩阵"""
    featureMatrix = np.hstack(
        [mean.reshape(2, 1), std.reshape(2, 1), rms, skew, kurs, papr, mmpc, wavent, dfa, hurst, petfd, sament,
         perent, hjorth, corpsd])

    return featureMatrix


def feature_save(path='F:\作业\本基项目\group'):
    """遍历所有.mat文件，映射到设计好的特征矩阵，并存储起来"""
    count = 0
    for rroot, rdirs, rfiles in os.walk(path, topdown=False):
        if len(rdirs) == 4:
            dirname = rroot.split('\\')[-1]
            for root, dirs, files in os.walk(rroot, topdown=False):
                if root != rroot:
                    filename = root.split('\\')[-1]
                    if filename != '4':
                        featuremat = getmat(root)
                        if not os.path.exists('F:\作业\本基项目\\featureMatrix\\' + dirname):
                            os.makedirs('F:\作业\本基项目\\featureMatrix\\' + dirname)
                        scipy.io.savemat('F:\作业\本基项目\\featureMatrix\\' + dirname + '\\' + filename + '.mat',
                                         {'feature': featuremat})
                        # print(count)
                        count += 1

    return count


def feature_mat(path):
    """将所有特征合并，并联合自我评估量表构成一个大型特征矩阵"""
    featureMatrix = np.array([])
    for root, dirs, files in os.walk(path, topdown=True):
        if root != path:
            # print(root)
            temp = np.array([])
            for file in files:
                data = scipy.io.loadmat(root + '\\' + file)
                temp = np.hstack((temp, np.array(data['feature']).reshape(-1)))
            if len(featureMatrix) == 0:
                featureMatrix = temp
            else:
                featureMatrix = np.vstack((featureMatrix, temp))
    data = DataFrame(featureMatrix)
    data.to_csv('F:\作业\本基项目\\feature_csv.csv')
    scipy.io.savemat('F:\作业\本基项目\\feature_mat.mat', {'feature': featureMatrix})

    return featureMatrix


if __name__ == "__main__":
    """
    读取数据，对相同情绪词性的trail做平均
    按编号存储特征矩阵，2（channel）*3024（feature）
    """
    feature_save()
    # featureMatrix = feature_mat('F:\作业\本基项目\\featureMatrix')
    # print(featureMatrix.shape)
    # feature = getmat('F:\作业\本基项目\group\\2021071416\\1')
    # print(feature.shape)
