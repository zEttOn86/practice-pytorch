# coding:utf-8
import os, sys, time
import torch
import torch.nn.functional as F
from model import Net
from dataset import get_data_loaders
from logger import create_summary_writer

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

if __name__ == "__main__":
    # Const
    epochs = 14
    train_batchsize = 64
    valid_batchsize = 1000
    log_interval = 10
    device = "cuda" if torch.cuda.is_available else "cpu"
    lr = 1.0
    log_dir = "./output"

    #
    model = Net()
    train_loader, val_loader = get_data_loaders(train_batchsize, valid_batchsize)
    writer = create_summary_writer(model, train_loader, log_dir)
    
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
    evaluator = create_supervised_evaluator(
                    model, metrics={"accuracy": Accuracy(), "nll": Loss(F.nll_loss)}, device=device
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(
            "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
            "".format(engine.state.epoch, engine.state.iteration, len(train_loader), engine.state.output)
        )
        writer.add_scalar("train/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        print(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_nll
            )
        )
        writer.add_scalar("train/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("train/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        print(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_nll
            )
        )
        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()
