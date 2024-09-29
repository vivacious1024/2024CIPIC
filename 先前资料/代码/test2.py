import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy.fft import fft
from scipy.signal import firwin, freqz, lfilter

from PCA import get_score

import FE
import PCA
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

featureMatrix = PCA.get_feature('F:\作业\本基项目\\feature_mat.mat')
# featureMatrix = (featureMatrix)/(featureMatrix.max(axis=0)-featureMatrix.min(axis=0))
featureMatrix = (featureMatrix-featureMatrix.min(axis=0))/(featureMatrix.max(axis=0)-featureMatrix.min(axis=0))
print(featureMatrix.shape)

score = get_score('F:\作业\本基项目\人格特质结果.xlsx')
print(score.shape)

bigfeature = np.hstack((featureMatrix,score))
print(bigfeature.shape)

pMatric =np.corrcoef(bigfeature.T)

# data = pd.DataFrame(pMatric)
#
# writer = pd.ExcelWriter('pMatric.xlsx')		# 写入Excel文件
# data.to_excel(writer, '20211222', float_format='%.5f')		# ‘20211222’是写入excel的sheet名
# writer.save()
#
# writer.close()
