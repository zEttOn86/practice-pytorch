import h5py
import numpy as np
from torch.utils.data import Dataset
# https://github.com/yjn870/SRCNN-pytorch/blob/064dbaac09859f5fa1b35608ab90145e2d60828b/datasets.py#L6

class SRDataset(Dataset):
    def __init__(self, h5_file):
        super(SRDataset, self).__init__()
        self.h5_file = h5_file
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)
    
    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

