import torch
import os
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from net import Animal_Net
from torchvision import transforms
from PIL import Image


label_dict = {'butterfly': 0,
              'cat': 1,
              'chicken': 2,
              'cow': 3,
              'dog': 4,
              'elephant': 5,
              'horse': 6,
              'sheep': 7,
              'spider': 8,
              'squirrel': 9}


class Animal_Test_Dataset(Dataset):
    def __init__(self, root,transform):
        self.dataset = []
        with open(root + '/test_label.txt', 'r') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                file_name, tag = lines[i].split(',')
                path = root + "/test/" + file_name
                self.dataset.append((path, tag[:-1]))
        self.transform = transform

    def __len__(self):
        # 返回数据集的长度
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        # 获取图像的路径
        path = data[0]
        # 获取图像的标签
        tag = data[1]
        # 利用PIL读取图片
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        iii= np.array(label_dict[tag]).astype(np.int64)
        iii = torch.from_numpy(iii)
        return img, iii
#图像预处理
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

if __name__ == '__main__':
    dataset_test = Animal_Test_Dataset('C:/Users/13291/Desktop/data',transform=test_transforms)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)
    net = Animal_Net()
    net.load_state_dict(torch.load("Animal-Classification.pth", map_location='cpu'))

    sum_score = 0.
    for i,(img, label) in enumerate(dataloader_test):
        net.eval()
        test_out = net(img)
        softmax = nn.Softmax(dim=1)
        test_out = softmax(test_out)
        pre = torch.argmax(test_out, dim=1)
        score = torch.mean(torch.eq(pre, label).float())
        sum_score = sum_score + score
        print(i)

    # 求平均分
    test_avg_score = sum_score / len(dataloader_test)

    print("测试得分：", test_avg_score)