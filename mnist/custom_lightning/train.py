# coding:utf-8
import os, sys, time
import argparse
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.logging.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import MNISTClassifier


if __name__ == "__main__":
    # Const
    parser  = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root_dir', default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--seed', type=int, default=2334)
    parser.add_argument('--gpus', type=int, default=0, help="how many gpus")
    parser.add_argument('--log_dir', default="./lightning_logs")
    parser.add_argument('--out_dir_name', default="{}".format(time.strftime('%Y-%m-%d_%H-%M-%S')))


    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--model_save_interval', type=int, default=1)
    parser.add_argument('--log_save_interval', type=int, default=100)
    parser.add_argument('--validation_interval', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', default=0.001)
    args = parser.parse_args()

    # init
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # To save
    # https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/731
    logger = TensorBoardLogger(
        save_dir = args.log_dir,
        name=args.out_dir_name
    )

    # https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
    checkpoint_callback = ModelCheckpoint(
        filepath = "{}/{}".format(args.log_dir, args.out_dir_name),
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