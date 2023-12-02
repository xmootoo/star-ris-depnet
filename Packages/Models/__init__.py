from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .model import WTNet
from .model_rp import WTRPNet

logger = getLogger()



def build_model(config) :
    """
    builds or reloads model and transfer them to config.device
    """
    if config.model_type.lower() == 'wtnet':
        model = WTNet(config)
    elif config.model_type.lower() == 'wtrpnet':
        model = WTRPNet(config)
    elif config.model_type.lower() == 'prnet':
        model = PRNet(config)
    else:
        assert False, 'Model type is not valid'
    if config.load_model:
        checkpoint = torch.load(config.model_path, map_location=torch.device(config.device))
        logger.info("============ Loading checkpoint ============")
        assert 'model' in list(checkpoint.keys())
        model.load_state_dict(checkpoint['model'])
        logger.info('============ Model loaded ============')
    model = model.to(config.device)
    return model
