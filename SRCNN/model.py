#coding:utf-8
import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl

from dataset import SRDataset

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5//2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5//2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)

class SRCNNTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super(SRCNNTrainer, self).__init__()
        self.net = SRCNN()
        self.cfg = cfg
        self.root_dir = cfg.datasets.root_dir
        
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        tensorboard_logs = {'train_loss' : loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.mse_loss(y_hat, y, reduction='mean')}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD([
            {'params': self.net.conv1.parameters()},
            {'params': self.net.conv2.parameters()},
            {'params': self.net.conv3.parameters(), 'lr':self.cfg.learning_rate * 0.1}
        ], lr=self.cfg.learning_rate)
        return optimizer
    
    @pl.data_loader
    def train_dataloader(self):
        data = SRDataset(os.path.join(self.root_dir, self.cfg.datasets.train_path))
        train_len = int(len(data)*0.8)
        val_len = int(len(data)-train_len)
        self.sr_train, self.sr_val = random_split(data, [train_len, val_len])
        return DataLoader(self.sr_train, batch_size=self.cfg.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.sr_val, batch_size=self.cfg.batch_size, shuffle=False)
    
    @pl.data_loader
    def test_dataloader(self):
        data = SRDataset(os.path.join(self.root_dir, self.cfg.datasets.test_path))
        return DataLoader(data, batch_size=self.cfg.batch_size, shuffle=False)