import numpy as np
import torch
import torch.nn
from lstm import LSTM

# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128,3,2,1),
            torch.nn.BatchNorm2d(128)
        )
        self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(16, 128,
                          kernel_size=1, stride=8, bias=False),
                torch.nn.BatchNorm2d(128),
            )
        self.relu = torch.nn.ReLU()
        # self.mlp1 = torch.nn.Linear(2*2*64,100)
        # self.mlp2 = torch.nn.Linear(100,10)
    def forward(self, x):
        x = self.conv1(x)
        res = self.downsample(x)
        print(res.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        print(x.shape)
        x += res
        x = self.relu(x)
#         x = self.mlp1(x.view(x.size(0),-1))
#         x = self.mlp2(x)
        return x

class CNN_LSTM(torch.nn.Module):
    def __init__(self, num_classes=5, batchsize=8):
        super(CNN_LSTM,self).__init__()
        self.cnn = CNNnet()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(128,3,3,1,1,bias=False),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU()
        )
        self.lstm = LSTM(seqlen=8, inputsize=39, hiddensize=num_classes, numlayer=3, batchsize=batchsize, droupout=0.2)
    
    def forward(self, x):
#         x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        x = self.lstm(x)*100
        return x