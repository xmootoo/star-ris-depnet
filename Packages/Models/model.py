import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
import os
import numpy as np
from .utils import *
from functools import reduce
import operator
from functorch import grad, vmap, hessian
from functools import partial
logger = getLogger()


class TNet_star(nn.Module):
    def __init__(self, config):
        super(TNet_star, self).__init__()
        self.config = config
        self.c1 = nn.Conv2d(2*config.K, 64, (4,4)).double()
        self.bn1 = nn.BatchNorm2d(64).double()
        self.c2 = nn.Conv2d(64, 64, (4,4)).double()
        self.bn2 = nn.BatchNorm2d(64).double()
        in_size = (config.N - 6)*(config.M - 6)*64
        self.l1 = nn.Linear(in_size, 4*config.N).double()
        self.bn3 = nn.BatchNorm1d(4*config.N).double()
        if config.star:
            self.l2 = nn.Linear(4*config.N, config.N).double()
            self.l3 = nn.Linear(4*config.N, config.N).double()
            self.l4 = nn.Linear(4*config.N, config.N).double()
        else:
            self.l2 = nn.Linear(4*config.N, config.N//2).double()
            self.l3 = nn.Linear(4*config.N, config.N//2).double()
        self.d1  = nn.Dropout(0.1)
        self.d2 = nn.Dropout(0.4)
        get_feature_stat(config)
    def forward(self, Data):
        da = get_features(self.config, Data, normalize = True)
        Input = da.reshape(da.shape[0], 2*self.config.K, self.config.M, self.config.N)
        temp = self.d1(F.relu(self.bn1(self.c1(Input))))
        temp = self.d1(F.relu(self.bn2(self.c2(temp))))
        temp = temp.reshape(temp.shape[0], -1)
        temp = self.d2(F.relu(self.bn3(self.l1(temp))))
        phi_t = self.l2(temp)
        phi_r = self.l3(temp)
        if self.config.star:
            a = self.l4(temp)
            return phi_t, phi_r, a
        else:
            return phi_t, phi_r

    def freaze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def unfreaze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True



class WNet(nn.Module):
    def __init__(self, config):
        super(WNet, self).__init__()
        self.config = config
        if config.star:
            self.l1 = nn.Linear(2*config.N*(config.K + config.M), 100).double()
        else:
            self.l1 = nn.Linear(config.N*(config.K + 2*config.M), 100).double()
        self.bn1 = nn.BatchNorm1d(100).double()
        self.l2 = nn.Linear(100, 100).double()
        self.bn2 = nn.BatchNorm1d(100).double()
        self.l3 = nn.Linear(100, 2*config.M*config.K + 1).double()
        self.d = nn.Dropout(0.1)
    def forward(self, Data):
        config = self.config
        H = Data['H']
        G = Data['G']
        H_bar = (H.real - config.mean['H'].real)/config.std['H'].real + 1j*(H.imag - config.mean['H'].imag)/config.std['H'].imag
        G_bar = (G.real - config.mean['G'].real)/config.std['G'].real + 1j*(G.imag - config.mean['G'].imag)/config.std['G'].imag
        if config.star:
            H_hat = torch.cat([H_bar, G_bar], dim = 2).unsqueeze(-1)
        else:
            H_hat = torch.cat([H_bar[:,:config.N//2,:2], H_bar[:,config.N//2:,2:], G_bar[:,:config.N//2,:], G_bar[:, config.N//2:,:]], dim = 2).unsqueeze(-1)
        Input = torch.cat([H_hat.real, H_hat.imag], dim = 3).reshape(H.shape[0], -1)
        temp = self.d(F.relu(self.bn1(self.l1(Input))))
        temp = self.d(F.relu(self.bn2(self.l2(temp))))
        temp = self.l3(temp)
        return temp



# def rate_vio(config, Data, w, phi_t, phi_r, **kwargs):
#     W, Theta_t, Theta_r = get_W_Theta(config, w, phi_t, phi_r, **kwargs)
#     R = rate(config, Data, Theta_t, Theta_r, W, users = True)
#     diff = F.relu(config.r_min - R + 1e-5)
#     return torch.linalg.norm(diff, ord = 2, dim = 1)**2

# def rate_vio_p(config, Data, w, phi_t, phi_r, **kwargs):
#     W, Theta_t, Theta_r = get_W_Theta(config, w, phi_t, phi_r, **kwargs)
#     R = rate(config, Data, Theta_t, Theta_r, W, users = True)
#     diff = config.p_gamma*(torch.exp(config.r_min - R) -1)
#     return diff.sum(-1)

# def rate_vio(config, Data, w, phi_t, phi_r, **kwargs):
#     W, Theta_t, Theta_r = get_W_Theta(config, w, phi_t, phi_r, **kwargs)
#     R = rate(config, Data, Theta_t, Theta_r, W, users = True)
#     diff = F.relu(config.r_min - R + 1e-5)
#     return torch.linalg.norm(diff, ord = 2, dim = 1)**2

# def rate_vio_p(config, Data, w, phi_t, phi_r, **kwargs):
#     W, Theta_t, Theta_r = get_W_Theta(config, w, phi_t, phi_r, **kwargs)
#     R = rate(config, Data, Theta_t, Theta_r, W, users = True)
#     diff = F.elu(torch.exp(config.r_min - R + 1e-5) -1)
#     return diff.sum(-1)

def rate_vio_p(config, Data, w, phi_t, phi_r, *args):
    W, Theta_t, Theta_r = get_W_Theta(config, w, phi_t, phi_r, *args)
    R = rate(config, Data, W, Theta_t, Theta_r, users = True)
    with torch.no_grad():
        x = F.relu(config.r_min - R)
        y = torch.linalg.norm(x, ord = 2, dim = 1)**2
        s = torch.heaviside(-(y-1e-3), values = torch.tensor(1).double().to(config.device))
    del x, W, Theta_t, Theta_r
    return (1-s)*((F.relu(torch.exp(config.r_min - R) -1)).sum(-1)) + s*((torch.exp(config.r_min - R) -1).sum(-1))

def V(config, Data, w, phi_t, phi_r, *args):
    return rate_vio_p(config, Data, w, phi_t, phi_r, *args).sum()



def project(config, params, Data, w, phi_t, phi_r, *args):
    grad_w_p = grad(V, argnums=2)
    grad_phi_t_p = grad(V, argnums = 3)
    grad_phi_r_p = grad(V, argnums = 4)
    if config.star:
        grad_a_p = grad(V, argnums = 5)

    old_step_w = 0
    old_step_phi_r = 0
    old_step_phi_t = 0
    if config.star:
        old_step_a = 0
        a = args[0]
    for i in range(params['num_iter']):

        gr_w = grad_w_p(config, Data, w, phi_t, phi_r, *args)
        gr_phi_t = grad_phi_t_p(config, Data, w, phi_t, phi_r, *args)
        gr_phi_r = grad_phi_r_p(config, Data, w, phi_t, phi_r, *args)
        if config.star:
            gr_a = grad_a_p(config, Data, w, phi_t, phi_r, a)

        new_step_w = params['lr_w']*gr_w + params['momentum_w']*old_step_w
        new_step_phi_t = params['lr_phi_t']*gr_phi_t + params['momentum_phi_t']*old_step_phi_t
        new_step_phi_r = params['lr_phi_r']*gr_phi_r + params['momentum_phi_r']*old_step_phi_r
        if config.star:
            new_step_a = params['lr_a']*gr_a + params['momentum_a']*old_step_a

        phi_r = phi_r - new_step_phi_r
        w = w - new_step_w
        phi_t = phi_t - new_step_phi_t
        if config.star:
            a = a - new_step_a

        old_step_w = new_step_w
        old_step_phi_r = new_step_phi_r
        old_step_phi_t = new_step_phi_t
        if config.star:
            old_step_a = new_step_a
            del gr_a, new_step_a
        del gr_w, gr_phi_r, gr_phi_t, new_step_phi_r, new_step_phi_t, new_step_w
    if config.star:
        return w, phi_t, phi_r, a
    else:
        return w, phi_t, phi_r

class WTNet(nn.Module):
    def __init__(self, config):
        super(WTNet, self).__init__()
        self.config = config
        self.tnet = TNet_star(config)
        self.wnet = WNet(config)
    def forward(self, Data):
        config = self.config
        if config.star:
            phi_t, phi_r, a = self.tnet(Data)
        else:
            phi_t, phi_r = self.tnet(Data)
        w = self.wnet(Data)
        if self.config.project:
            p_params = self.config.train_params if self.training else self.config.test_params
            if config.star:
                w_p, phi_t_p, phi_r_p, a_p = project(config, p_params, Data, w, phi_t, phi_r, a)
                return w_p, phi_t_p, phi_r_p, a_p
            else:
                w_p, phi_t_p, phi_r_p = project(config, p_params, Data, w, phi_t, phi_r)
                return w_p, phi_t_p, phi_r_p
        else:
            if config.star:
                return w, phi_t, phi_r, a
            else:
                return w, phi_t, phi_r
