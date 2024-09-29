import numpy as np
import torch
import torch.nn
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from resnet import ResNet
from lstm import RNN, LSTM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ResLSTM(nn.Module):
    def __init__(self, depth=38, num_classes=5, batchsize=8):
        super(ResLSTM, self).__init__()
        self.resnet = ResNet(depth, num_classes)
        self.conv1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(10, 5)
        self.sigmoid = nn.Sigmoid()
        self.lstm = LSTM(seqlen=76, inputsize=256, hiddensize=128, numlayer=1, batchsize=batchsize)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.resnet(out)
        print(out)
        # out = torch.cat((out1, out2), dim=1)
        # out = self.fc(out)
        # out = self.sigmoid(out)
        # out = torch.cat((out1, out2), dim=1)
        # out = out.permute(0, 2, 1)
        # print(out)
        out = out.permute(0, 3, 1, 2)
        out = out.contiguous().view(out.size(0), out.size(1), -1)
        # temp = temp.contiguous().view(out.size(0), 1, -1)
        # for i in range(1, out.shape[2]):
        #     temp1 = out[:, :, i, :].contiguous().view(out.size(0), 1, -1)
        #     temp = torch.cat((temp, temp1), dim=1)
        # print(temp.shape)  # [batch, 38, 2048]
        # out = (out-torch.min(out))/(torch.max(out)-torch.min(out))
        out = (out-torch.mean(out))/torch.std(out)
        # print(out)
        out = self.lstm(out)
        # print(out)
        out = self.fc1(out)
        out = self.tanh(out/100.0)
        # print(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)*100
        # print(out)
        return out


def tensor_shape(tensor):
    # 将emedding后的两个通道数据合在一个维度中
    result = torch.cat((tensor[0, 0, :, :], tensor[0, 1, :, :]), 1)[np.newaxis, :, :]
    for i in range(1, tensor.shape[0]):
        temp = torch.cat((tensor[i, 0, :, :], tensor[i, 1, :, :]), 1)[np.newaxis, :, :]
        result = torch.cat((result, temp), 0)

    return result[:, np.newaxis, :, :]


if __name__ == "__main__":
    data = torch.randint(0, 1000000, (100, 2, 610))
    print(data.shape)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=20)

    embedding = nn.Embedding(1000000, 200)
    for batch, x in enumerate(dataloader):
        print(batch)
        print(x[0].shape)
        out = embedding(x[0])
        print(out.shape)
        result = torch.cat((out[0, 0, :, :], out[0, 1, :, :]), 1)[np.newaxis, :, :]
        for i in range(1, out.shape[0]):
            temp = torch.cat((out[i, 0, :, :], out[i, 1, :, :]), 1)[np.newaxis, :, :]
            result = torch.cat((result, temp), 0)
        print(result.shape)
    test = torch.tensor([1., np.nan, 2])
    print(torch.isnan(test).any())
