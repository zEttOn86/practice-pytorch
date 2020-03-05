# coding:utf-8
import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms

import pytorch_lightning as pl
import hydra

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(self.dropout2(x))
        return F.log_softmax(x, dim=1)

class MNISTClassifier(pl.LightningModule):
    """
    https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
    https://github.com/PyTorchLightning/pytorch-lightning-conference-seed/blob/master/research_seed/mnist/mnist.py
    """
    def __init__(self, hparams):
        super(MNISTClassifier, self).__init__()
        self.net = Net()
        self.hparams = hparams
        self.root_dir = hydra.utils.get_original_cwd()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.nll_loss(y_hat, y)}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=self.hparams.learning_rate)
    
    @pl.data_loader
    def train_dataloader(self):
        data_transform = transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,))])

        mnist_train = MNIST(download=True, \
                                root=os.path.join(self.root_dir,"../../data"), \
                                train=True, \
                                transform=data_transform)
        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        return DataLoader(self.mnist_train, batch_size=self.hparams.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.hparams.batch_size, shuffle=False)

    @pl.data_loader
    def test_dataloader(self):
        data_transform = transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,))])
        mnist_test = MNIST(download=True, \
                                root=os.path.join(self.root_dir,"../../data"), \
                                train=False, \
                                transform=data_transform)
        return DataLoader(mnist_test, batch_size=self.hparams.batch_size, shuffle=False)


