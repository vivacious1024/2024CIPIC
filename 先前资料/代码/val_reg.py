import matplotlib.pyplot as plt
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
epochs = 1
batch_size = 256
num_workers = 4


dataset = Dataset_eeg("F:\作业\本基项目\\psdcom")
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)


# val
def val():
    # model = torch.load('ResNet_Model\CNN_LSTM_psdcom_best.pt').to(device)
    model = CNN_LSTM(num_classes=5, batchsize=batch_size)
    model.load_state_dict(torch.load('ResNet_Model/CNN_LSTM_psdcom_state.pt', map_location="cuda:0"))
    model = model.to(device)
    print(model)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    model.eval()
    label_test = []
    pred_test = []
    with torch.no_grad():
        for batch, [data, label] in enumerate(val_loader):
            data = data.float().to(device)
            label = label.float().to(device)

            predict = model.forward(data)
            # model.eval()
            predict = predict.view(-1).data.cpu().numpy().tolist()
            # print(label)
            # print(pred_test)
            label_test.extend(label.view(-1).data.cpu().numpy().tolist())
            pred_test.extend(predict)
    val_r2 = r2_score(label_test, pred_test)
    val_mse = mean_squared_error(pred_test, label_test)
    val_rmse = np.sqrt(val_mse)
    print("val:   r2:", val_r2)
    print("val:   mse:", val_mse)
    print("val:   rmse:", val_rmse)
    length = len(label_test)
    print('-------ex-------')
    print("r2:", r2_score([label_test[x] for x in np.arange(0,length-1,5)],
                          [pred_test[x] for x in np.arange(0,length-1,5)]))
    print("mse:", mean_squared_error([label_test[x] for x in np.arange(0, length - 1, 5)],
                          [pred_test[x] for x in np.arange(0, length - 1, 5)]))
    print("rmse:", np.sqrt(mean_squared_error([label_test[x] for x in np.arange(0, length - 1, 5)],
                          [pred_test[x] for x in np.arange(0, length - 1, 5)])))
    print('-------ag-------')
    print("r2:", r2_score([label_test[x] for x in np.arange(1, length - 1, 5)],
                          [pred_test[x] for x in np.arange(1, length - 1, 5)]))
    print("mse:", mean_squared_error([label_test[x] for x in np.arange(1, length - 1, 5)],
                                     [pred_test[x] for x in np.arange(1, length - 1, 5)]))
    print("rmse:", np.sqrt(mean_squared_error([label_test[x] for x in np.arange(1, length - 1, 5)],
                                              [pred_test[x] for x in np.arange(1, length - 1, 5)])))
    print('-------co-------')
    print("r2:", r2_score([label_test[x] for x in np.arange(2, length - 1, 5)],
                          [pred_test[x] for x in np.arange(2, length - 1, 5)]))
    print("mse:", mean_squared_error([label_test[x] for x in np.arange(2, length - 1, 5)],
                                     [pred_test[x] for x in np.arange(2, length - 1, 5)]))
    print("rmse:", np.sqrt(mean_squared_error([label_test[x] for x in np.arange(2, length - 1, 5)],
                                              [pred_test[x] for x in np.arange(2, length - 1, 5)])))
    print('-------ne-------')
    print("r2:", r2_score([label_test[x] for x in np.arange(3, length - 1, 5)],
                          [pred_test[x] for x in np.arange(3, length - 1, 5)]))
    print("mse:", mean_squared_error([label_test[x] for x in np.arange(3, length - 1, 5)],
                                     [pred_test[x] for x in np.arange(3, length - 1, 5)]))
    print("rmse:", np.sqrt(mean_squared_error([label_test[x] for x in np.arange(3, length - 1, 5)],
                                              [pred_test[x] for x in np.arange(3, length - 1, 5)])))
    print('-------op-------')
    print("r2:", r2_score([label_test[x] for x in np.arange(4, length - 1, 5)],
                          [pred_test[x] for x in np.arange(4, length - 1, 5)]))
    print("mse:", mean_squared_error([label_test[x] for x in np.arange(4, length - 1, 5)],
                                     [pred_test[x] for x in np.arange(4, length - 1, 5)]))
    print("rmse:", np.sqrt(mean_squared_error([label_test[x] for x in np.arange(4, length - 1, 5)],
                                              [pred_test[x] for x in np.arange(4, length - 1, 5)])))
    # print('label:', label_test)
    # print('pred_test', pred_test)

    plt.scatter(label_test, pred_test)
    plt.show()


if __name__ == "__main__":
    val()

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
