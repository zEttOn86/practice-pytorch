# coding:utf-8
import os, sys, time
from torch.utils.tensorboard import SummaryWriter

def create_summary_writer(model, data_loader, log_dir):
    """
    https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist_with_tensorboardx.py
    """
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer