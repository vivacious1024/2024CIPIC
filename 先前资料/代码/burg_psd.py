import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift
import scipy.signal

# 自定义自相关函数
import FE


def xcorr(data):
    length = len(data)
    R = []
    for m in range(0, length):
        sum = 0.0
        for n in range(0, length - m):
            sum = sum + data[n] * data[n + m]
        R.append(sum / length)
    return R


N = int(input("请输入采样点数: "))
p = int(input("请选择AR模型阶数p(N/3<p<N/2):"))
# 构造原函数
f1 = 100
f2 = 15
sample=500
time = np.arange(0,N)/sample
print(time)
print(time.shape)

xn = 8 * np.sin(2 * np.pi * f1 * time + np.pi / 3) + 10 * np.cos(2 * np.pi * f2 * time + np.pi / 4)
# 加入高斯白噪声
wn = np.random.normal(0, 1, len(xn))
yn = []
for x, y in zip(xn, wn):
    z = x + y
    yn.append(z)

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
print(w.shape, h.shape)
Hf = abs(h)
Sx = Hf ** 2
f = (w / (2 * np.pi)) * sample
print(f)

freq = fftfreq(N, d=1/sample)
print(freq)
plt.subplot(311)
plt.plot(time, R)
plt.subplot(312)
plt.plot(f, Sx)
plt.title("power spectrum")
plt.xlabel("f")
plt.ylabel("Sx")
# plt.show()

'''返回峰值索引，幅值降序'''


def findpeak(arr):
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


# 估计fl,f2
# s = int((f1+f2)*N)
# f1_guji =(np.argmax(Sx[0:s]))/(2*N)
# f2_guji= (np.argmax(Sx[s:int(N / 2)])+s)/(2*N)
# print("f1 的估计值:f1=", f1_guji)
# print("f2 的估计值:f2=", f2_guji)
f1_guji = findpeak(Sx)[0] * (sample/2/N)
f2_guji = findpeak(Sx)[1] * (sample/2/N)
print("f1 的估计值:f1=", f1_guji)
print("f2 的估计值:f2=", f2_guji)

# 自相关谱估计
num_fft = N
cor_x = np.correlate(yn, yn, 'same')
cor_x = fft(cor_x, num_fft)
ps_cor = np.abs(cor_x)
ps_cor = ps_cor / np.max(ps_cor)
ax = plt.subplot(313)
ax.set_title('indirect method')
plt.plot(freq[0:int(N / 2)], 20 * np.log10(ps_cor[:num_fft//2]))
plt.show()
