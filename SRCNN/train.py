# coding:utf-8
import os, sys, time
import torch
import numpy as np
import hydra

from pytorch_lightning import Trainer
from pytorch_lightning.logging.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import SRCNNTrainer
from six import string_types
@hydra.main(config_path='./configs/config.yaml')
def main(cfg):
    # Reset seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    cfg.log_dir = os.getcwd()
    # Set logger
    logger = TensorBoardLogger(
        save_dir = cfg.log_dir
    )

    # Set checkpoint
    checkpoint_callback = ModelCheckpoint(
        filepath = "{}".format(cfg.log_dir),
        save_top_k=-1, 
        verbose=True,
        monitor="val_loss",
        mode="min",
        period=cfg.model_save_interval
    )
    
    if cfg.datasets.root_dir == "None":
        cfg.datasets.root_dir = hydra.utils.get_original_cwd()
    print(cfg.pretty())
    
    model = SRCNNTrainer(cfg)

    trainer = Trainer(
        gpus=cfg.gpus,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=cfg.epochs,
        log_save_interval=cfg.log_save_interval,
        check_val_every_n_epoch=cfg.validation_interval
    )

    # Start training
    trainer.fit(model)


if __name__ == "__main__":
    main()