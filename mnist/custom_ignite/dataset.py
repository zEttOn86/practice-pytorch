#coding:utf-8
import os, sys, time
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import datasets, transforms

def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(MNIST(download=True, 
                                    root="../../data",
                                    train=True,
                                    transform=data_transform),
                            batch_size=train_batch_size, 
                            shuffle=True,
                            num_workers=1)
    
    val_loader = DataLoader(MNIST(download=False,
                                  root="../../data",
                                  train=False,
                                  transform=data_transform),
                            batch_size=val_batch_size, 
                            shuffle=False, 
                            num_workers=1)

    return train_loader, val_loader
