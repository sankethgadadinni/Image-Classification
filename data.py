import os
import cv2
import torch
import pandas as pd
import torchvision
from torch.nn import functional as F
from torch import nn
import torchvision.transforms as transforms

from random import randint
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


base_dir = 'Sanketh/Image-Classification/DATASET'
train_path = 'Sanketh/Image-Classification/DATASET/TRAIN'
test_path = 'Sanketh/Image-Classification/DATASET/TEST'

classes_dir_data = os.listdir(train_path)
num = 0
classes_dict = {}
num_dict = {}
for c in  classes_dir_data:
    classes_dict[c] = num
    num_dict[num] = c
    num = num +1


img_size = 50
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.005),
    transforms.Grayscale(1),
    transforms.ToTensor()])


def target_transform(x):
    return torch.tensor(classes_dict.get(x))



class ImageDataset(Dataset):
    def __init__(self, classes, base_dir, transform=None, target_transform=None):

        
        self.base_dir = base_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = classes
                
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        
        img_dir_list = os.listdir(os.path.join(self.base_dir,self.img_labels[index]))
        
        image_path = img_dir_list[randint(0,len(img_dir_list)-1)]

        image_path = os.path.join(self.base_dir,self.img_labels[index],image_path)

        image = cv2.imread(image_path)

        if self.transform:

            image = self.transform(image)

        if self.transform:

            label = self.target_transform(self.img_labels[index])

        return image,label
    


class YogaDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
    
    
    def prepare_data(self):
        
        self.train = ImageDataset(classes_dir_data,train_path,train_transform,target_transform)

        self.valid = ImageDataset(classes_dir_data,test_path,train_transform,target_transform)

        self.test = ImageDataset(classes_dir_data,test_path,train_transform,target_transform)
        
    
    def train_dataloader(self):
    
        return DataLoader(self.train,batch_size = self.batch_size,shuffle = True)

    def val_dataloader(self):  

        return DataLoader(self.valid,batch_size = self.batch_size,shuffle = True)

    def test_dataloader(self):

        return DataLoader(self.test,batch_size = self.batch_size,shuffle = True)
            
        
    