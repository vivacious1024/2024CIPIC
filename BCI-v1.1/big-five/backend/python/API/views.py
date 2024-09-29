from django.http import HttpResponse

import scipy
import pywt
import numpy as np
import joblib
import json
import pandas as pd

from . import eeg2mat

def getFeatures(signal):
  coeffs = pywt.wavedec(signal, 'db1', level=3, mode='periodic')
  cA3, cD3, cD2, cD1 = coeffs
  y = pywt.waverec(coeffs, 'db1', mode='periodic')

  features = [
    sum(cD1) / len(cD1),
    sum(cD2) / len(cD2),
    sum(cD3) / len(cD3),
    sum(cA3) / len(cA3)
  ]

  return features

def sum_abs(signal):
  ans = 0.0
  for i in signal:
    ans += abs(i)
  return ans

with open('../../../config.json', 'r') as file:
    path0 = json.load(file) 
with open('../../../user.json', 'r') as file:
    path1 = json.load(file) 
eegPath = path0+'/'+path1+'/data'

def load_csv(path):
  data_read = pd.read_csv(path)
  list = data_read.values.tolist()
  data = np.array(list)
  print(data.shape)
  return data


def hello(request):
  token = request.GET.get('token')

  print(token)
  

  with open(eegPath + '/' + token + '/data.eeg', mode='rb') as file:
    eegsrc = file.read()
    eeg = eeg2mat.batchData(eegsrc)
    scipy.io.savemat(eegPath + '/' + token + '/data.mat', { 'eegdata': eeg['data'] })

  obj = scipy.io.loadmat(eegPath + '/' + token + '/data.mat')
  ch1 = obj['eegdata'][0][0][0][0]
  ch2 = obj['eegdata'][0][0][1][0]

  createTime = 0
  stopTime = 0

  f_createTime = open(eegPath + '/' + token + '/createTime.txt', 'r')
  createTime = int(f_createTime.readline())
  f_createTime.close()

  f_logs = open(eegPath + '/' + token + '/log.txt', 'r')
  logs = f_logs.read()
  f_logs.close()

  behavior = []
  logs = logs.split('\n')

  seq = []
  endstamps = []

  for j in range(len(logs)):
    item = logs[j].split(',')
    if(len(item) >= 3):
      if(item[2] == 'stop'):
        stopTime = int(item[0])
      elif(item[2].startswith('word')):
        if(item[3] == 'start'):
          seq.append(item[2].split('=')[1])
        elif(item[3] == 'end'):
          endstamps.append(int(item[0]))
      elif(item[2].startswith('resolveKey=Space') and len(seq) > 0):
        endstamps.append(int(item[0]))

  print(len(ch1), len(ch2))
  fout = open(eegPath + '/' + token + '/data.txt', 'w')
  fout.write('avg_ch1,avg_ch2,cD1_ch1,cD2_ch1,cD3_ch1,cA3_ch1,cD1_ch2,cD2_ch2,cD3_ch2,cA3_ch2,word_id\n')
  print('seq', seq)
  for j in range(len(seq)):
    word_id = int(seq[j])
    stime = endstamps[j] - 1200
    diff = (stime - createTime) // 2
    print('diff:', diff)
    sub_ch1 = ch1[diff:diff+600] # 如果要降采样在这里搞
    sub_ch2 = ch2[diff:diff+600]
    print(len(sub_ch1), len(sub_ch2))
    if(len(sub_ch1) == 0 or len(sub_ch2) == 0):
      break
    features1_1 = getFeatures(sub_ch1)
    features1_2 = getFeatures(sub_ch2)
    fout.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s\n' % (
      sum_abs(sub_ch1) / 5000,
      sum_abs(sub_ch2) / 5000,
      features1_1[0],
      features1_1[1],
      features1_1[2],
      features1_1[3],
      features1_2[0],
      features1_2[1],
      features1_2[2],
      features1_2[3],
      word_id
    ))
  fout.close()

  data = load_csv(eegPath + '/' + token + '/data.txt')
  feat = data[:, 0:11]

  # model.predict
  model_a = joblib.load('./models/a.pkl')
  model_c = joblib.load('./models/c.pkl')
  model_e = joblib.load('./models/e.pkl')
  model_n = joblib.load('./models/n.pkl')
  model_o = joblib.load('./models/o.pkl')

  print(feat)
  print(feat.shape)

  pred_o = model_o.predict(feat)
  pred_c = model_c.predict(feat)
  pred_e = model_e.predict(feat)
  pred_a = model_a.predict(feat)
  pred_n = model_n.predict(feat)

  score_o = 0
  score_c = 0
  score_e = 0
  score_a = 0
  score_n = 0

  for i in range(len(pred_o)):
    if(pred_o[i] == 'T'):
      score_o += 1
  score_o /= len(pred_o)

  for i in range(len(pred_c)):
    if(pred_c[i] == 'T'):
      score_c += 1
  score_c /= len(pred_c)

  for i in range(len(pred_e)):
    if(pred_e[i] == 'T'):
      score_e += 1
  score_e /= len(pred_e)

  for i in range(len(pred_a)):
    if(pred_a[i] == 'T'):
      score_a += 1
  score_a /= len(pred_a)

  for i in range(len(pred_n)):
    if(pred_n[i] == 'T'):
      score_n += 1
  score_n /= len(pred_n)

  # score_o = sum(model_o.predict(dataList)) / len(model_o.predict(dataList))
  # score_c = sum(model_c.predict(dataList)) / len(model_c.predict(dataList))
  # score_e = sum(model_e.predict(dataList)) / len(model_e.predict(dataList))
  # score_a = sum(model_a.predict(dataList)) / len(model_a.predict(dataList))
  # score_n = sum(model_n.predict(dataList)) / len(model_n.predict(dataList))

  # 用这里1/0的评分去做回归

  # 专注度计算
  # score_con = 0
  # for i in range(180, 200):
  #   if data.get(str(i+1)):
  #     dat = data.get(str(i+1))
  #     if dat.get('ch1') and dat.get('ch2'):
  #       score_con += len(dat.get('ch1'))

  # score_con /= 20
  # score_con = 1 - score_con / 1000

  result = {
    'score_o': score_o,
    'score_c': score_c,
    'score_e': score_e,
    'score_a': score_a,
    'score_n': score_n,
    'score_con': 1
  }
  print(result)

  return HttpResponse(json.dumps(result))
