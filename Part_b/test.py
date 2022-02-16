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

    test_data = sys.argv[1]
    test_dataset = ImageDataset(data_csv=test_data,train=False,img_transform=img_transforms)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)

    mpth = sys.argv[2]
    predic = sys.argv[3]

    model = Net()
    model.load_state_dict(torch.load(mpth))
    model.eval()
    ans = np.empty(shape=(0,),dtype=int)
    with torch.no_grad():
        for batch_idx,sample in enumerate(test_loader):
            img = sample['images']
            images = img.cuda()
            mo = model.cuda()
            output = mo(images)
            _,pred = output.max(1)
            pred = pred.cpu()
            narr = pred.numpy()
            ans = np.concatenate((ans,narr),axis=0)
    np.savetxt(predic,ans)

