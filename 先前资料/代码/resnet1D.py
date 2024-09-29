import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim

from resnet import BottleNeck


class BottleNeck1D(nn.Module):
    '''
    使用1x1卷积核减少参数量 提高训练速度
    '''
    expansion = 4

    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BottleNeck1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, outplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(outplanes)
        self.conv2 = nn.Conv1d(outplanes, outplanes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(outplanes)
        self.conv3 = nn.Conv1d(outplanes, outplanes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(outplanes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            res = self.downsample(x)

        out += res
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, depth=38, num_classes=5):
        """
        :param x:
        :return:
        """
        super(ResNet1D, self).__init__()
        n = (depth - 2) // 9
        self.inplanes = 16
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16)
        )
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, n)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        self.avgpool = nn.AvgPool1d(4)
        # self.fc1 = nn.Linear(64 * BottleNeck1D.expansion, num_classes)
        self.fc1 = nn.Linear(144, num_classes)

    def _make_layer(self, planes, n, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BottleNeck1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * BottleNeck1D.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * BottleNeck1D.expansion),
            )
        layers = []
        layers.append(BottleNeck1D(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BottleNeck1D.expansion
        for i in range(1, n):
            layers.append(BottleNeck1D(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        # x = self.fc1(x)
        return x


if __name__ == '__main__':
    batchsz = 32

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    # model = Lenet5().to(device)
    model = ResNet1D().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(1000):

        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #
        print(epoch, 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            print(epoch, 'acc:', acc)
