import numpy as np
import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter   
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms as T
from sklearn.metrics import r2_score, mean_squared_error

from dataloader_v2 import Dataset_eeg, Dataset_eeg2
from lstm import RNN, LSTM
from resnet import ResNet
from resnet_lstm import ResLSTM
from cnntest import CNN_LSTM


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
train_size = 0.8
epochs = 200
batch_size = 256
num_workers = 4
lr = 0.01
writer = SummaryWriter('../../data-output')

dataset = Dataset_eeg("psdcom")
train_dataset = []
test_dataset = []
# 划分训练集和测试集
for i in range(int(train_size * len(dataset))):
    train_dataset.append(dataset[i])
for i in range(int(train_size * len(dataset)), len(dataset)):
    test_dataset.append(dataset[i])

# train_dataset2 = Dataset_eeg2(train_dataset, transform=T.Compose([T.Normalize(0, 1)]))
train_dataset2 = Dataset_eeg2(train_dataset)
test_dataset2 = Dataset_eeg2(test_dataset)
train_loader = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_dataset2, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


# train
def train():
#     model = ResLSTM(depth=18, batchsize=batch_size).to(device)
    model = CNN_LSTM(num_classes=5, batchsize=batch_size).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    # 参数寻优，计算损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, threshold=0.01)
    loss_func = torch.nn.MSELoss().to(device)

    for epoch in range(epochs):
        print("--------------%d / %d epochs---------------" % (epoch + 1, epochs))
        model.train()
        label_train = []
        pred_train = []
        for batchidx, [data, label] in enumerate(train_loader):
            # print(data.shape)
            data = data.float().to(device)
            label = label.float().to(device)
            # torch.backends.cudnn.enabled = False
            predict = model(data)
            print('label:',label,'\npredict:',predict)
            label_train.extend(label.view(-1).data.cpu().numpy().tolist())
            pred_train.extend(predict.view(-1).data.cpu().numpy().tolist())
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = loss_func(predict, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        r2 = r2_score(label_train,pred_train)
        mse = mean_squared_error(pred_train, label_train)
        rmse = np.sqrt(mse)
        print("train:   r2:", r2, " loss:", loss.item())
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/r2', r2, epoch)
        writer.add_scalar('train/MSE', mse, epoch)
        writer.add_scalar('train/RMSE', rmse, epoch)

        model.eval()
        label_test = []
        pred_test = []
        with torch.no_grad():
            for batch, [data, label] in enumerate(test_loader):
                data = data.float().to(device)
                label = label.float().to(device)

                predict = model(data)
                predict = predict.view(-1).data.cpu().numpy().tolist()
                # print(label)
                # print(pred_test)
                label_test.extend(label.view(-1).data.cpu().numpy().tolist())
                pred_test.extend(predict)
        val_r2 = r2_score(label_test, pred_test)
        val_loss = loss_func(torch.tensor(pred_test), torch.tensor(label_test))
        val_mse = mean_squared_error(pred_test, label_test)
        val_rmse = np.sqrt(val_mse)
        scheduler.step(val_loss)
        print("test:   r2:", val_r2, " loss:", val_loss.item(), "  lr:", optimizer.param_groups[0]['lr'])
        writer.add_scalar('test/loss', val_loss, epoch)
        writer.add_scalar('test/r2', val_r2, epoch)
        writer.add_scalar('test/MSE', val_mse, epoch)
        writer.add_scalar('test/RMSE', val_rmse, epoch)
        
#         print(pred_train)

    # 保存模型
    torch.save(model, 'ResNet_Model/resnet.pt')

if __name__ == "__main__":
    train()

# train_size = int(len(dataX) * 0.7)
#
# x_train = dataX[:train_size]  # 训练数据
# y_train = dataY[:train_size]  # 训练数据目标值
#
# x_train = x_train.reshape(-1, 1, 2)  # 将训练数据调整成pytorch中lstm算法的输入维度
# y_train = y_train.reshape(-1, 1, 1)  # 将目标值调整成pytorch中lstm算法的输出维度
#
# # 将ndarray数据转换为张量，因为pytorch用的数据类型是张量
# x_train = torch.from_numpy(x_train)
# y_train = torch.from_numpy(y_train)
#
# rnn = RNN().to(device)
# # 参数寻优，计算损失函数
# optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
# loss_func = torch.nn.MSELoss()
#
# for i in range(1000):
#     var_x = Variable(x_train).type(torch.FloatTensor).to(device)
#     var_y = Variable(y_train).type(torch.FloatTensor).to(device)
#     out = rnn(var_x)
#     loss = loss_func(out, var_y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if (i + 1) % 100 == 0:
#         print('Epoch:{}, Loss:{:.5f}'.format(i + 1, loss.item()))
#
# # 准备测试数据
# dataX1 = dataX.reshape(-1, 1, 2)
# dataX2 = torch.from_numpy(dataX1)
# var_dataX = Variable(dataX2).type(torch.FloatTensor).to(device)
#
# rnn.eval()
# pred = rnn(var_dataX)
# pred_test = pred.view(-1).data.cpu().numpy()  # 转换成一维的ndarray数据，这是预测值
# print(pred_test.shape)
# print(pred_test)
#
# plt.plot(pred_test, 'r', label='prediction')
# plt.plot(dataY, 'b', label='real')
# plt.legend(loc='best')
# plt.show()
