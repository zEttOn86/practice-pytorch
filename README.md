## Practice pytorch

  
### Consturct pytorch environment

OS   : Windows10  
CUDA : 8.0  
  
1. Update cuda version  
- [CUDA Toolkit 10.1 update2 Archive](https://developer.nvidia.com/cuda-10.1-download-archive-update2)
  
2. Install cudnn. First of all, download cudnn and unzip it, finally, put it into cuda version directory.
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

3. Install pytorch and tensorboard  
- [pytorch](https://pytorch.org/)
- [TORCH.UTILS.TENSORBOARD](https://pytorch.org/docs/stable/tensorboard.html)


### Reference
- [pytorch/pytorch](https://github.com/pytorch/pytorch)
- [pytorch/examples](https://github.com/pytorch/examples)
- [pytorch/tutorials](https://github.com/pytorch/tutorials)


### How to open tensorboard
To run the TensorBoard, open a new terminal and run the command below. Then, open http://localhost:6006/ on your web browser.

```
tensorboard --logdir='./logs' --port=6006
```

- [pytorch-tutorial/tutorials/04-utils/tensorboard/](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard)
