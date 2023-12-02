import torch
import torch.nn.functional as F
from .comlib import *
import numpy as np
from functorch import grad, vmap, hessian
from functools import partial


def loss_n(config, Data, W, Theta_t, Theta_r):

    R = rate(config, Data, W, Theta_t, Theta_r, users = True)
    r = R.sum(-1)
    # diff = F.relu(config.r_min - R)
    # vio = torch.linalg.norm(diff, ord = 2, dim = 1)**2
    with torch.no_grad():
        x = F.relu(config.r_min - R)
        y = torch.linalg.norm(x, ord = 2, dim = 1)**2
        s = torch.heaviside(-(y-1e-3), values = torch.tensor(1).double().to(config.device))
    vio = config.gamma*((1-s)*((F.relu(torch.exp(config.r_min - R) -1)).sum(-1))) + s*((torch.exp(config.r_min - R) -1).sum(-1))

    # diff = F.elu(torch.exp(config.r_min - R + 1e-5) -1)
    # vio = diff.sum(-1)
    score = -(r) + vio
    return score.mean()

def loss_rp(config, Data, var_t, var_r, w, r_min, p_max):

    W, Theta_t, Theta_r = get_W_Theta(config, w, var_t, var_r, mode1 = config.mode1)
    W_p = (W/config.P_max)*p_max.reshape(W.shape[0], 1, 1)
    R = rate(config, Data, Theta_t, Theta_r, W, users = True)
    R_p = rate(config, Data, Theta_t, Theta_r, W_p, users = True)
    r = R.sum(-1)
    # diff = F.relu(r_min - R_p + 1e-5)
    diff = F.elu(torch.exp(r_min - R + 1e-5) -1)
    vio = diff.sum(-1)
    # vio = torch.linalg.norm(diff, ord = 2, dim = 1)**2
    score = -1*(r) + config.gamma*vio
    return score.mean()
