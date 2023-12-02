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
        self.l2 = nn.Linear(4*config.N, config.N).double()
        self.l3 = nn.Linear(4*config.N, config.N).double()
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
        a = self.l3(temp)
        return phi_t, a

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

    def freaze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def unfreaze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True


class RNet(nn.Module):
    def __init__(self, config):
        super(RNet, self).__init__()
        self.config = config
        if config.star:
            self.l1 = nn.Linear(2*config.N*(config.K + config.M), 100).double()
        else:
            self.l1 = nn.Linear(config.N*(config.K + 2*config.M), 100).double()
        self.bn1 = nn.BatchNorm1d(100).double()
        self.l2 = nn.Linear(100, 100).double()
        self.bn2 = nn.BatchNorm1d(100).double()
        self.l3 = nn.Linear(100, config.K).double()
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
        r_min = F.relu(temp) + config.r_min
        return r_min

    def freaze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def unfreaze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True



class PNet(nn.Module):
    def __init__(self, config):
        super(PNet, self).__init__()
        self.config = config
        if config.star:
            self.l1 = nn.Linear(2*config.N*(config.K + config.M), 100).double()
        else:
            self.l1 = nn.Linear(config.N*(config.K + 2*config.M), 100).double()
        self.bn1 = nn.BatchNorm1d(100).double()
        self.l2 = nn.Linear(100, 100).double()
        self.bn2 = nn.BatchNorm1d(100).double()
        self.l3 = nn.Linear(100, 1).double()
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
        p_max = torch.sigmoid(temp + config.p_bias)*config.P_max
        return p_max

    def freaze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def unfreaze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True



def rate_vio(config, Data, var_t, var_r, w, r_min, p_max, **kwargs):
    W, Theta_t, Theta_r = get_W_Theta(config, w, var_t, var_r, p_max = p_max, **kwargs)
    R = rate(config, Data, Theta_t, Theta_r, W, users = True)
    diff = F.relu(r_min - R + 1e-5)
    return torch.linalg.norm(diff, ord = 2, dim = 1)**2
def V(config, Data, var_t, var_r, w, r_min, p_max, **kwargs):
    return rate_vio(config, Data, var_t, var_r, w, r_min, p_max, **kwargs).sum()

grad_w_p = grad(V, argnums=4)
grad_var_t_p = grad(V, argnums = 2)
grad_var_r_p = grad(V, argnums = 3)

def project(config, params, Data, var_t, var_r, w, r_min, p_max):
    old_step_w = 0
    old_step_var_r = 0
    old_step_var_t = 0
    for i in range(params['num_iter']):

        gr_w = grad_w_p(config, Data, var_t, var_r, w, r_min, p_max, mode1 = config.mode1)
        gr_var_t = grad_var_t_p(config, Data, var_t, var_r, w, r_min, p_max, mode1 = config.mode1)
        gr_var_r = grad_var_r_p(config, Data, var_t, var_r, w, r_min, p_max, mode1 = config.mode1)

        new_step_w = params['lr_w']*gr_w + params['momentum_w']*old_step_w
        new_step_var_t = params['lr_var_t']*gr_var_t + params['momentum_var_t']*old_step_var_t
        new_step_var_r = params['lr_var_r']*gr_var_r + params['momentum_var_r']*old_step_var_r
        var_r = var_r - new_step_var_r
        w = w - new_step_w
        var_t = var_t - new_step_var_t

        old_step_w = new_step_w
        old_step_var_r = new_step_var_r
        old_step_var_t = new_step_var_t

    return w, var_t, var_r

class WTRPNet(nn.Module):
    def __init__(self, config):
        super(WTRPNet, self).__init__()
        assert config.star, 'DNN is only for the star mode'
        self.config = config
        self.tnet = TNet_star(config)
        self.wnet = WNet(config)
        self.rnet = RNet(config)
        self.pnet = PNet(config)
        self.nets = {'t': self.tnet, 'w': self.wnet, 'p': self.pnet, 'r': self.rnet}
    def forward(self, Data, init):
        config = self.config
        var_t, var_r = self.tnet(Data)
        w = self.wnet(Data)
        if 'r' in init:
            r_min = config.r_min*torch.ones(w.shape[0], config.K).double().to(config.device)
        else:
            r_min = self.rnet(Data)
        if 'p' in init:
            p_max = config.P_max*torch.ones(config.batch_size).double().to(config.device)
        else:
            p_max = self.pnet(Data)

        if self.config.project:
            if self.training:
                w, var_t, var_r = project(self.config, self.config.train_params, Data, var_t, var_r, w, r_min, p_max)
            else:
                w, var_t, var_r = project(self.config, self.config.test_params, Data, var_t, var_r, w, r_min, p_max)
        return w, var_t, var_r, r_min, p_max

    def adjust(self, active_dnns):
        assert set(active_dnns).issubset(set(self.nets)), 'Wrong set of dnns!!'
        for k,v in self.nets.items():
            if k in active_dnns:
                v.unfreaze()
                v.train()
            else:
                v.freaze()
                v.eval()
