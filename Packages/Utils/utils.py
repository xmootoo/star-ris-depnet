import pandas as pd
import os
import random
import math
import torch
import numpy as np
import json
import argparse
from logging import getLogger
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import shutil
import pickle
# from ..Models import build_model
from .config import *
from .datahandler import create_data_loader
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import time
from .comlib import *
from .objective import *

logger = getLogger()
FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

def set_seed(config):
    """Set seed"""
    if config.env_seed == -1 :
        config.env_seed = np.random.randint(1_000_000_000)
    seed = config.env_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f'seed set to {config.env_seed}')

def to_cuda(config, *args):
    """
    Move tensors to CUDA.
    """
    if config.device.type == 'cpu':
        return args
    return [None if x is None else x.to(config.device) for x in args]
