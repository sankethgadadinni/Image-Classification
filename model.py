import os
from pickletools import optimize
import torch
import cv2
import pandas as pd
from torch.nn import functional as F
from torch import nn
import torchvision.transforms as transforms

from omegaconf import OmegaConf
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.core.lightning import LightningModule
from data import YogaDataModule
from network import Net


class YogaClassificationTask(LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.net = Net(config=config)
        
    
    def step(self, batch, batch_idx):
        
        x, y = batch
        y_pred = self.net(x)
        loss = F.cross_entropy(y_pred, y)
        return loss
    
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val", loss)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss
    
    
    def configure_optimizers(self):
        lr = self.config.learning_rate
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        return optimizer
    
    
    def predict(self,image_path):
        image = cv2.imread(image_path)


        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((50, 50)),
            transforms.Grayscale(1),
            transforms.ToTensor()])   
        
        image = train_transform(image)
        image = image.view(-1,1,50,50)
        preds = self.net(image)
        probs = torch.softmax(preds, dim=1)
        out = probs.argmax(dim=1)
        return out 


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="provide the config file")
    args = parser.parse_args()

    configFile = OmegaConf.load(args.config)
    config = configFile.config

    tb_logger = pl_loggers.TensorBoardLogger('logs/YogaClassification')

    dm = YogaDataModule(config=config)

    model = YogaClassificationTask(config=config)

    trainer = pl.Trainer(max_epochs=config.epochs,gpus=config.gpus,log_every_n_steps=1, progress_bar_refresh_rate=20,logger=tb_logger)

    trainer.fit(model, dm)
    trainer.test(model, dm)
    
