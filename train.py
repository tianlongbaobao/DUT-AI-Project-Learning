import os
from tqdm import tqdm
import torch
import torch.optim as optim
from torchvision import transforms,  datasets
import numpy as np
from net import Animal_Net
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import time
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
#路径保存
train_dir="../data/train"
test_dir="../data/test"
label_dir="../data/test_label.txt"
#基础参数
batch_size = 64
device = 'cuda'
epochs = 50
#数据预处理
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
#训练数据集
train_datasets=datasets.ImageFolder(train_dir,transform=train_transforms)
train_dataloaders= DataLoader(train_datasets,batch_size=batch_size,shuffle=True)
train_dataset_sizes=len(train_datasets)

#test数据集
class MyDataset(Dataset):
    def __init__(self, img_path,label_path, transform):
        super(MyDataset, self).__init__()
        self.root = img_path
        self.txt_root = label_path
        f = open(self.txt_root, 'r')
        data = f.readlines()
        imgs = []
        labels = []
        for line in data:
            line = line.rstrip()
            word = line.split(",")
            imgs.append(os.path.join(self.root, word[0]))
            if word[1]=="butterfly":
                join_=0
            elif word[1]=="cat":
                join_=1
            elif word[1]=="chicken":
                join_=2
            elif word[1]=="cow":
                join_=3
            elif word[1]=="dog":
                join_=4
            elif word[1]=="elephant":
                join_=5
            elif word[1]=="horse":
                join_=6
            elif word[1]=="sheep":
                join_=7
            elif word[1]=="spider":
                join_=8
            elif word[1]=="squirrel":
                join_=9
            labels.append(join_)
        self.img = imgs
        self.label = labels
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return img, label
test_datasets=MyDataset(test_dir,label_dir,transform=test_transforms)
test_dataloaders= DataLoader(test_datasets,batch_size=batch_size,shuffle=True)
test_dataset_sizes=len(test_datasets)
# 训练模型
# 定义网络
net = Animal_Net()
net.to(device)
loss_function = nn.CrossEntropyLoss()
# train
best_acc=0.0
loss=0.0
start_time = time.time()
a=[]
b=[]

epoch_list=[]
for epoch in range(epochs):
    if epoch <= 10 :
        lr = 0.1
    elif epoch <= 20:
        lr = 0.01
    elif epoch <= 30:
        lr = 0.001
    elif epoch <=40 :
        lr = 0.0005
    else:
        lr = 0.0001

    optimizer = optim.SGD(net.parameters(), lr, weight_decay=0.0001,momentum=0.9)
    net.train()
    running_loss = 0.0
    #训练集开始
    for images, labels in tqdm(train_dataloaders):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    #验证集开始
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for x, y in tqdm(test_dataloaders):
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            predicted = torch.max(outputs, dim=1)[1]
            acc += (predicted == y).sum().item()
    #数据整合处理
    accurate = acc / test_dataset_sizes
    train_loss = running_loss / train_dataset_sizes
    a.append(accurate)
    b.append(train_loss)
    epoch_list.append(epoch+1)
    #画成图像更为直观
    fig, ax = plt.subplots()
    ax.plot(epoch_list, a, label='accuracy')
    ax.plot(epoch_list, b, label='train loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('rate')
    ax.set_title('training statics')
    ax.legend()
    plt.show()
    print('[epoch %d] train_loss: %.3f   accuracy: %.3f' %
              (epoch + 1, train_loss, accurate))
    #如果没有上一次准确率好则不保存
    if best_acc<=accurate:
       best_acc=accurate
       loss=train_loss
       torch.save(net.state_dict(), "Animal-Classification2.pth")
       print(f"此模型准确率{best_acc},损失值{train_loss}")
#模型评价与保存
print(f"训练完成，此模型准确率{best_acc},损失值{loss}")
end_time = time.time()
o=(end_time-start_time)/60
print("共用时%.3f分钟" % o)