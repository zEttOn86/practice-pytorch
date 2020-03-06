# Super Resolution Convolutional Neural Network (SRCNN)
This pytorch implementation is based on [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092).

### Requirement
- pytorch 1.4
- hydra-core 0.11.3
- pytorch-lightning 0.6.0

### Dataset
Download from [here](https://github.com/yjn870/SRCNN-pytorch/tree/064dbaac09859f5fa1b35608ab90145e2d60828b).
  
I use 91-image, which scale is 4, for training and validation.
And I use Set5 for test.

### How to run
Training

```bash
python train.py
```

Test (Under construction)
```bash
python test.py
```

### Reference
- [yjn870/SRCNN-pytorch](https://github.com/yjn870/SRCNN-pytorch/tree/064dbaac09859f5fa1b35608ab90145e2d60828b)