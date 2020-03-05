# coding:utf-8
import os, sys, time
import argparse
import torch
import numpy as np
import hydra

from pytorch_lightning import Trainer
from pytorch_lightning.logging.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import MNISTClassifier


@hydra.main(config_path='./config/config.yaml')
def main(cfg):
    args = cfg
    # init
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    print(type(args))
    args.log_dir = os.getcwd()
    # To save
    # https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/731
    logger = TensorBoardLogger(
        save_dir = os.getcwd()
    )

    # https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
    checkpoint_callback = ModelCheckpoint(
        filepath = "{}".format(args.log_dir),
        save_top_k=-1, 
        verbose=True,
        monitor="val_loss",
        mode="min",
        period=args.model_save_interval
    )

    model = MNISTClassifier(args)
    trainer = Trainer(
        gpus=args.gpus,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=args.epochs,
        log_save_interval=args.log_save_interval,
        check_val_every_n_epoch=args.validation_interval
    )

    # Start training
    trainer.fit(model)

if __name__ == "__main__":
    main()