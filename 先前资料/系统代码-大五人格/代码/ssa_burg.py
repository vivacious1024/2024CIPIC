#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import FE

sample_rate = 500
sig_size = 500
x = np.arange(sig_size) / sample_rate
series = np.sin(2 * np.pi * x) + 5 * np.cos(7 * np.pi * x) + 3 * np.random.random(len(x))
series = series - np.mean(series)  # 中心化(非必须)

# step1 嵌入
windowLen = 200  # 嵌入窗口长度
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
    mainfre = FE.myburg(U[:, i], sample_rate)
    if mainfre <= (sample_rate / 4):
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

rrr = np.sum(rec[0:5,:], axis=0)  # 选择重构的部分，这里选了前5个
r_low = np.sum(rec[U_low,:],axis=0)
r_high = np.sum(rec[U_high,:],axis=0)
plt.figure()
for i in range(12):
    ax = plt.subplot(6, 2, i + 1)
    ax.plot(x, rec[i, :])

plt.figure(2)
plt.subplot(221)
plt.plot(x, series)
plt.subplot(222)
plt.plot(x, rrr)
plt.subplot(223)
plt.plot(x, r_low)
plt.subplot(224)
plt.plot(x, r_high)
plt.show()
