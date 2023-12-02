import pandas as pd
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from logging import getLogger
import math
import itertools
import torch
from shutil import copyfile
import copy
import sys

from main import get_parser, main
from Packages import *
from Packages.Utils import utils as ut
from Packages.Utils import data_generation as dg

if __name__ == "main":

        max_epoch = str(sys.argv[1])

        # Step 1: Experiment & Learning Parameters 
        model_path = '/home/mehrazin/Documents/NGWN/JUPNet/Pre_Trained'
        arg = []

        # experiment parameters
        arg += ['--exp_id', '1112', # ID number of the folder to save models and logs (saved to 'Dumped' folder)
                '--env_seed', '1',
                '--load_model', 'False',
                '--eval_only', 'False',
                '--model_path', model_path,
                '--model_type', 'WTNet',
                '--dataset_id', '17', # Dataset that you will use to train on.
                '--data_preparation', 'False',
                '--debug', 'False',
                '--debug_id', '1',
                '--debug_train_samples', '2',
                '--debug_test_samples', '2',
                '--debug_common_samples', '2',
                '--debug_seed', '2',
                '--normalize', 'True',
                '--augment', 'False',
                '--device', 'cpu',
                '--pre_augment', 'False']

        # learning hyperparameters
        arg += ['--clip_grad_norm', '0',
                '--learning_rate', '0.0001',
                '--decay_rate', '0.99',
                '--weight_decay', '0.05',
                '--max_epoch', max_epoch, # Number of epochs to train for
                '--epoch_size', '10000', # Number of examples used at each epoch (randomly sampled)
                '--batch_size', '100',
                '--optimizer', 'Adam']

        args = get_parser()
        args = args.parse_args(arg)


        # Step 2: Model Parameters
        config = Config(args)
        config.project = True
        config.proj_on = 1
        config.gamma = 1
        for k, v in [('lr_w', 0.1), ('lr_phi_t', 0.1), ('lr_phi_r', 0.1), ('lr_a', 0.1), ('momentum_var_t', 0.5), ('momentum_var_r', 0.5), ('momentum_w', 0.5), ('momentum_a', 0.5), ('num_iter', 100)]:
                config.train_params[k] = v
                config.test_params[k] = v
        config.train_params['num_iter'] = 10
        config.test_params['num_iter'] = 20
        config.train_increase = 0
        config.r_min = 3 # Configure for the dataset
        config.mode1 = True
        config.star = True # Select STAR-RIS or Normal RIS
        config.model_id = 1
        config.active_dnns = ['r']
        config.init = []
        config.p_bias = 2
        config.P_max = 2 # Configure for the dataset
        config.N = 64 # Configure for the dataset (default is 64, unless using special datasets)
        N0_dbm = -170
        config.N0 = (10**((N0_dbm - 30)/10))*180*1e3

        config.check_name = 'checkpoint.pth'
        config.eval_dtypes = ['test', 'valid']
        for feature in ['exp_dir', 'device', 'model_type', 'normalize', 'epoch_size', 'N0', 'M', 'P_max']:
        print(getattr(config, feature))


        # Step 3: Run NN
        main(config)