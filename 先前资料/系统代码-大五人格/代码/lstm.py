import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # 输入shape(seq_len,batch,input_size)
        # 如果batch_first=True,则输入shape(batch,seq_len,input_size)
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=4,
            num_layers=5,
            batch_first=True,
        )
        self.hidden = (torch.autograd.Variable(torch.zeros(5, 128, 4)).to(device), torch.autograd.Variable(torch.zeros(5, 128, 4)).to(device))
        self.linear = nn.Linear(4, 1)
        self.fc= nn.Linear(610,5)

    def forward(self, x):
        r_out, self.hidden = self.lstm(x, self.hidden)  # ？？？
        # print('r_out:', r_out.shape)
        # print('hn:', self.hidden[0].shape, '\ncn:', self.hidden[1].shape)
        self.hidden=(Variable(self.hidden[0]),Variable(self.hidden[1]))#可以把这一步去掉，在loss.backward（）中加retain_graph=True，主要是Varible有记忆功能，而张量没有
        outs = []  # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):  # 对每一个时间点计算 output
            outs.append(self.linear(r_out[:, time_step, :]))
        # print(outs)
        outs = catlist(outs)
        # outs = torch.stack(outs,dim=1)
        # print(outs.shape)
        outs = self.fc(outs)
        return outs

class LSTM(nn.Module):
    def __init__(self, seqlen=610, inputsize=20, hiddensize=64, numlayer=5, batchfirst=True, numclasses=50, batchsize=8, bidirectional=False, droupout=0.2):
        super(LSTM, self).__init__()

        # 输入shape(seq_len,batch,input_size)
        # 如果batch_first=True,则输入shape(batch,seq_len,input_size)
        self.lstm = nn.LSTM(
            input_size=inputsize,
            hidden_size=hiddensize,
            num_layers=numlayer,
            batch_first=batchfirst,
            bidirectional=bidirectional,
            dropout=droupout
        )
        num_directions = 2 if bidirectional else 1
        # self.h0 = torch.zeros(num_directions*numlayer, batchsize, hiddensize, dtype=torch.float32).requires_grad_().to(device)  # num_layers*num_directions, batch, hidden_size
        # self.c0 = torch.zeros(num_directions*numlayer, batchsize, hiddensize, dtype=torch.float32).requires_grad_().to(device)
        # self.linear = nn.Linear(num_directions*hiddensize, numclasses)  # 最后一个时刻的隐藏节点全连接
        # self.fc= nn.Linear(seqlen,numclasses)  # 序列轴向全连接

    def forward(self, x):
        # h0 = torch.zeros(1, 8, 128).requires_grad_().to(device)
        # c0 = torch.zeros(1, 8, 128).requires_grad_().to(device)
        # r_out, _ = self.lstm(x, (self.h0, self.c0))  # ？？？
        r_out, _ = self.lstm(x)  # ？？？
        # print('r_out:', r_out.shape)
        # print('hn:', self.hidden[0].shape, '\ncn:', self.hidden[1].shape)
        # out = self.linear(r_out[:, -1, :])
        out = r_out[:, -1, :]
#         out = self.linear(out)
        # print(out)
        return out


# 数据集和目标值赋值，dataset为数据，look_back为以几行数据为特征维度数量
def creat_dataset(dataset, look_back):
    data_x = []
    data_y = []
    for i in range(len(dataset) - look_back):
        data_x.append(dataset[i:i + look_back])
        data_y.append(dataset[i + look_back])
    return np.asarray(data_x), np.asarray(data_y)  # 转为ndarray数据

# 将tensor列表转成一整个tensor
def catlist(ls):
    if len(ls)!=0:
        result = ls[0]
    for i in range(1, len(ls)):
        result = torch.cat((result, ls[i]), 1)
    return result


if __name__ == "__main__":
    time = np.linspace(0, 100, 100 * 25)
    data = np.sin(time) + 0.1*np.random.randn(100 * 25) + 55
    f = 2000
    print('总数据长度', data.shape)
    # 归一化处理，这一步必不可少，不然后面训练数据误差会很大，模型没法用
    max_value = np.max(data)
    min_value = np.min(data)
    scalar = max_value - min_value
    datas = list(map(lambda x: (x-min_value) / scalar, data))

    # 以2为特征维度，得到数据集
    dataX, dataY = creat_dataset(datas, f)
    print('自变量：', dataX.shape)

    train_size = int(len(dataX) * 0.7)

    x_train = dataX[:train_size]  # 训练数据
    y_train = dataY[:train_size]  # 训练数据目标值

    x_train = x_train.reshape(-1, 1, f)  # 将训练数据调整成pytorch中lstm算法的输入维度
    # x_train = x_train[:,:,np.newaxis]
    y_train = y_train.reshape(-1, 1)  # 将目标值调整成pytorch中lstm算法的输出维度

    # 将ndarray数据转换为张量，因为pytorch用的数据类型是张量
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    rnn = LSTM(seqlen=1, inputsize=f,hiddensize=8,numlayer=1,numclasses=1,batchsize=350).to(device)
    # 参数寻优，计算损失函数
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    for i in range(1000):
        var_x = Variable(x_train).type(torch.FloatTensor).to(device)
        var_y = Variable(y_train).type(torch.FloatTensor).to(device)
        out = rnn(var_x)
        loss = loss_func(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('Epoch:{}, Loss:{:.5f}'.format(i + 1, loss.item()))

    # 准备测试数据
    dataX1 = dataX.reshape(-1, 1, f)
    dataX2 = torch.from_numpy(dataX1)
    var_dataX = Variable(dataX2).type(torch.FloatTensor).to(device)

    rnn.eval()
    pred = rnn(var_x)
    pred_test = pred.view(-1).data.cpu().numpy()  # 转换成一维的ndarray数据，这是预测值
    print(pred_test.shape)
    print(dataY)
    print(pred_test)

    plt.plot(pred_test, 'r', label='prediction')
    plt.plot(dataY, 'b', label='real')
    plt.legend(loc='best')
    plt.show()
