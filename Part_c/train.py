import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import sys

class ImageDataset(Dataset):
    def __init__(self, data_csv,train=True,img_transform=None):
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        data = pd.read_csv(data_csv,header=None)
        if self.is_train:
            images = data.iloc[:,1:].to_numpy()
            labels = data.iloc[:,0].astype(int)
        else:
            images = data.iloc[:,:].to_numpy()
            labels = None
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape((32,32,3),order="F")
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        image = self.img_transform(image)
        sample = {"images": image, "labels": label}
        return sample

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=(3,3),padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.bn12 = nn.BatchNorm2d(32)
        self.bn13 = nn.BatchNorm2d(32)
        self.conv12 = nn.Conv2d(32,32,kernel_size=(3,3),padding='same')
        self.conv13 = nn.Conv2d(32,32,kernel_size=(3,3),padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.drp1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=(3,3),padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.bn22 = nn.BatchNorm2d(64)
        self.bn23 = nn.BatchNorm2d(64)
        self.conv22 = nn.Conv2d(64,64,kernel_size=(3,3),padding='same')
        self.conv23 = nn.Conv2d(64,64,kernel_size=(3,3),padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.drp2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv2d(64,128,kernel_size=(3,3),padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.bn32 = nn.BatchNorm2d(128)
        self.bn33 = nn.BatchNorm2d(128)
        self.conv32 = nn.Conv2d(128,128,kernel_size=(3,3),padding='same')
        self.conv33 = nn.Conv2d(128,128,kernel_size=(3,3),padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.drp3 = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(2048,512)
        self.bnfc1 = nn.BatchNorm1d(512)
        self.drpf4 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512,256)
        self.bnfc2 = nn.BatchNorm1d(256)
        self.drpf5 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256,10)

    def forward(self,X):
        X = F.elu(self.conv1(X))
        X = self.bn1(X)
        X = F.elu(self.conv12(X))
        X = self.bn12(X)
        X = F.elu(self.conv13(X))
        X = self.bn13(X)
        X = self.pool1(X)
        X = self.drp1(X)
        X = F.elu(self.conv2(X))
        X = self.bn2(X)
        X = F.elu(self.conv22(X))
        X = self.bn22(X)
        X = F.elu(self.conv23(X))
        X = self.bn23(X)
        X = self.pool2(X)
        X = self.drp2(X)
        X = F.elu(self.conv3(X))
        X = self.bn3(X)
        X = F.elu(self.conv32(X))
        X = self.bn32(X)
        X = F.elu(self.conv33(X))
        X = self.bn33(X)
        X = self.pool3(X)
        X = self.drp3(X)
        X = X.view(-1,2048)
        X = F.elu(self.fc1(X))
        X = self.bnfc1(X)
        X = self.drpf4(X)
        X = F.elu(self.fc2(X))
        X = self.bnfc2(X)
        X = self.drpf5(X)
        X = self.fc3(X)
        return X

if __name__ == '__main__':
    torch.manual_seed(51)
    BATCH_SIZE = 64
    NUM_WORKERS = 20
    img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    img_transforms1 = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    train_data = sys.argv[1]
    train_dataset = ImageDataset(data_csv=train_data,train=True,img_transform=img_transforms)
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    test_data = sys.argv[2]
    test_dataset = ImageDataset(data_csv=test_data,train=True,img_transform=img_transforms)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)

    mpth = sys.argv[3]
    locloss = sys.argv[4]
    locacc = sys.argv[5]
    arrl = []
    arrac = []

    model = Net()
    nump = sum(p.numel() for p in model.parameters())
    crit = nn.CrossEntropyLoss()
    opti = optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    bestacc = -1
    epochs = 50
    i = 1

    while(i <= epochs):
        if(i == 15):
            opti = optim.Adam(model.parameters(),lr=0.0005,weight_decay=0.0001)
        if(i == 25):
            opti = optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.0001)
        if(i == 40):
            opti = optim.Adam(model.parameters(),lr=0.00005,weight_decay=0.0001)
        num = 0
        lsum = 0
        for batch_idx,sample in enumerate(train_loader):
            img = sample['images']
            lab = sample['labels']
            images = img.cuda()
            labels = lab.cuda()
            model = model.cuda()
            crit = crit.cuda()
            output = model(images)
            labels = labels.long()
            loss = crit(output,labels)
            opti.zero_grad()
            loss.backward()
            opti.step()
            lsum = lsum+loss.item()
            num = num+1
        lavg = lsum/num
        arrl.append(lavg)
        model.eval()
        ncorr = 0
        nsam = 0
        with torch.no_grad():
            for batch_idx,sample in enumerate(test_loader):
                images = sample['images']
                labels = sample['labels']
                images = images.cuda()
                labels = labels.cuda()
                mo = model.cuda()
                output = mo(images)
                _,pred = output.max(1)
                nsam = nsam+labels.size(0)
                ncorr = ncorr+pred.eq(labels).sum().item()

        accr = ncorr/nsam
        arrac.append(accr)

        if(accr >= bestacc):
            torch.save(model.state_dict(),mpth)
            with open(locloss,'w') as f:
                for item in arrl:
                    f.write("%s\n"%item)
            with open(locacc,'w') as f:
                for item in arrac:
                    f.write("%s\n"%item)
            bestacc = accr

        model.train()
        i = i+1

