import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2

dataset_dir = './dataset/45_training/ROI/train/'
gmm_dir = './dataset/45_training/ROI/gmm_28x28/'

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ParkingLotDataset():
    def __init__(self, transform=None):
        self.dataset_dir = dataset_dir
        self.gmm_dir = gmm_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.dataset_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.dataset_dir)[idx]
        img_idx = img_name.split('.')
        img_idx = img_idx[0][3:-1]
        img = cv2.imread(self.dataset_dir + img_name)  # input image
        img = cv2.resize(img, (224, 224))
        
        gmm = cv2.imread(self.gmm_dir + img_name, cv2.IMREAD_GRAYSCALE)
        gmm = np.expand_dims(gmm, axis=0)

        if self.transform:
            img = self.transform(img)
            # gmm = self.transform(gmm)

        item = {'I': img, 'X': img_idx, 'G': gmm}

        return item


pl_dataset = {x: ParkingLotDataset(transform=data_transforms[x])
              for x in ['train']}
pl_dataloader = {x: DataLoader(pl_dataset[x], batch_size=4, shuffle=True, num_workers=4)
                 for x in ['train']}
