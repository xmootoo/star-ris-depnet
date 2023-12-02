import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, BatchSampler
from scipy.io import loadmat, savemat
import itertools
from tqdm import tqdm
from .data_generation import *

class Data(Dataset):
    """
    This class creates an interface for the dataset for the DataLoader function
    """
    def __init__(self, config, dtype):
        super(Data, self).__init__()
        self.dtype = dtype
        self.path = config.norm_data
        self.load_data(config)
        self.get_stats(config)
        if dtype == 'train':
            self.path = config.train_dir
        if dtype == 'test':
            self.path = config.test_dir
        if dtype == 'valid':
            self.path = config.valid_dir
        self.load_data(config)
        self.config = config

    def load_data(self, config):
        """
        Loads the data from the disk and put to self.data
        """

        data_path = os.path.join(self.path, 'data.mat')
        temp = loadmat(data_path)
        G, H = temp['G'], temp['H']
        self.size = G.shape[0]
        self.data = {'G': torch.from_numpy(G), 'H': torch.from_numpy(H)}

    def __len__(self):
        """
        Retuens the size of the dataset
        """
        return self.size

    def __getitem__(self, idx):
        """
        Returns an item in index = idx from the dataset
        """
        item = {}
        for k in ['G', 'H']:
            item[k] = self.data[k][idx, :, :]
        return item

    def get_stats(self, config):
        config.mean = {}
        config.std = {}
        for k in ['G', 'H']:
            config.mean[k] = self.data[k].real.mean(0).to(config.device) + 1j*self.data[k].imag.mean(0).to(config.device)
            config.std[k] = self.data[k].real.std(0).to(config.device) + 1j*self.data[k].imag.std(0).to(config.device)

    def collate_fn(self, elements):
        """
        Make batches by data augmentation
        """
        batch = {}
        b_size = len(elements)
        for k in ['G', 'H']:
            batch[k] = torch.zeros((b_size, elements[0][k].shape[0], elements[0][k].shape[1])).double() + 1j*torch.zeros((b_size, elements[0][k].shape[0], elements[0][k].shape[1])).double()
        for i, item in enumerate(elements):
            for k in ['G', 'H']:
                batch[k][i, :, :] = item[k]
        return batch


def create_data_loader(config, dtype):
    """
    Returns an iterable to iterate over the data. dtype can be one of the followings:
    'train', 'test'
    """
    assert dtype in ['train', 'test', 'valid']
    dataset = Data(config, dtype)
    if dtype in ['train']:
        shuffle = True
    else:
        shuffle = False
    df = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle
    )
    return df
