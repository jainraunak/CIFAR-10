import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import sys


class ImageDataset(Dataset):
    def __init__(self, data_csv, train=True, img_transform=None):
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
        self.conv1 = nn.Conv2d(3,32,kernel_size=(3,3),stride=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(32,64,kernel_size=(3,3),stride=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv3 = nn.Conv2d(64,512,kernel_size=(3,3),stride=(1,1))
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv4 = nn.Conv2d(512,1024,kernel_size=(2,2),stride=(1,1))
        self.fc1 = nn.Linear(1024,256)
        self.drp = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256,10)

    def forward(self,X):
        X = self.conv1(X)
        X = F.relu(self.bn1(X))
        X = self.pool1(X)
        X = self.conv2(X)
        X = F.relu(self.bn2(X))
        X = self.pool2(X)
        X = self.conv3(X)
        X = F.relu(self.bn3(X))
        X = self.pool3(X)
        X = self.conv4(X)
        X = F.relu(X)
        X = X.view(-1,1024)
        X = F.relu(self.fc1(X))
        X = self.drp(X)
        X = self.fc2(X)
        return X

if __name__ == '__main__':
    torch.manual_seed(51)
    BATCH_SIZE = 200
    NUM_WORKERS = 20
    img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
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
    crit = nn.CrossEntropyLoss()
    opti = optim.Adam(model.parameters(),lr=0.0001)
    epochs = 5
    i = 1
    while(i <= epochs):
        num = 0
        lsum = 0
        for batch_idx,sample in enumerate(train_loader):
            img = sample['images']
            lab = sample['labels']
            images = img.cuda()
            labels = lab.cuda()
            mo = model.cuda()
            crit = crit.cuda()
            output = mo(images)
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
        torch.save(model.state_dict(),mpth)
        model.train()
        accr = ncorr/nsam
        arrac.append(accr)
        i = i+1

    torch.save(model.state_dict(),mpth)
    with open(locloss,'w') as f:
        for item in arrl:
            f.write("%s\n"%item)

    with open(locacc,'w') as f:
        for item in arrac:
            f.write("%s\n"%item)
