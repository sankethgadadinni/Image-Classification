import os
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


class Net(nn.Module):
    def __init__(self, config) -> None:
        super(Net, self).__init__()
        
        self.config = config
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.linear1 = nn.Linear(12*50*50, 1000)    ###The input dimensions are(No. of dim * height * width)
        self.linear2 = nn.Linear(1000, 50)
        self.linear3 = nn.Linear(50, 10)
        self.linear4 = nn.Linear(10, 5)
        
    def forward(self, input):
        
        out = self.conv1(input)
        out = self.pool(out)
        
        out = self.conv2(out)
        out = self.pool(out)
        
        out = self.conv3(out)
        out = self.pool(out)
        
        out = out.view(out.size(0), -1)
        
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        
        return out

        
        
        
        
        
