import h5py
import matplotlib.pyplot
import numpy as np
import scipy
import torch
from torch.utils import data
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
import math
from torch.utils.data import DataLoader
import time
from torchvision.utils import save_image
import PCA


class Dataset_eeg(data.Dataset):
    def __init__(self, root, transforms=None):
        '''img_folders = os.listdir(root)
        # print(img_folders)
        self.imgs = []
        for img_folder in img_folders:
            images = os.listdir(root+'/'+img_folder)
            # print(imgs)
            self.imgs = self.imgs + ([os.path.join(root+'/'+img_folder, img) for img in images])
        '''
        self.data = []
        self.label = []
        isample = 0  # 第i个样本
        iword = 0  # 第i个词性
        for foldername in os.listdir(root):
            # print(foldername) #just for test
            labelpath = "F:/作业/本基项目/bigfive_1121.xlsx"
            score = PCA.get_score(labelpath)
            isample += 1
            for wordfolder in os.listdir(root + "/" + foldername):
                # img is used to store the image data
                # print(r"./" + root + "/" + foldername + "/" + filename)
                if wordfolder != "4":
                    WordRoot = root + "/" + foldername + "/" + wordfolder
                else:
                    continue
                for filename in os.listdir(WordRoot):
                    FileRoot = WordRoot + "/" + filename
                    self.data.append(FileRoot)
                    self.label.append(list(score[isample - 1, :]))
                # print(FileRoot)
        # print(self.data)
        index = np.arange(len(self.data))
        np.random.shuffle(index)
        self.data = np.array(self.data)[index]
        self.data = self.data.tolist()
        self.label = np.array(self.label)[index]
        self.label = self.label.tolist()
        self.transforms = transforms
        # print(self.label)
        # print(self.imgs)

    def __getitem__(self, index):
        data_path = self.data[index]
        # label = math.floor(index / 100)
        # print(img_path.split('/')[-2])
        label = np.array(self.label[index])
        label = torch.from_numpy(label)
        # data = scipy.io.loadmat(data_path)
        data = h5py.File(data_path)
        # data = np.reshape(data['epoch'], (20, 61))[np.newaxis, :, :]
        # data = (data['epoch']-np.min(data['epoch']))/(np.max(data['epoch'])-np.min(data['epoch']))#归一化
        data = data['epoch']
        datanp = np.empty((59,3000))
        for i in range(data.shape[0]):
            datanp[i,:] = (data[i,:] - np.mean(data[i,:])) / np.std(data[i,:])
        data = datanp*10
        # data = datanp
        plt.plot(np.repeat(np.arange(0,6000,2).reshape(1,3000),59, axis=0),data)
        plt.show()
        data = torch.from_numpy(data)
        if self.transforms:
            data = self.transforms(data)
        # save_image(data, './save/'+str(index)+'.png')  # 为0返回图片数据, save_img能将Tensor保存成图片
        return data, label

    def __len__(self):
        return len(self.data)


class Dataset_eeg2(data.Dataset):
    def __init__(self,dataset,transform=None):
        self.dataset=dataset
        self.transform=transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        data,label=self.dataset[idx]
        return data,label


if __name__ == "__main__":
    dataset = Dataset_eeg("F:/作业/本基项目/group")
    dataset2 = Dataset_eeg2(dataset)
    dataloder = DataLoader(dataset2, batch_size=2, shuffle=True)
    print(dataloder)
    for batch, [data, label] in enumerate(dataloder):
        print(data.shape)
        # print(label)
        # if data[0].shape == (2, 2, 610):
        #     print(data)  # 输出为空，shape都为(2, 2, 610)
