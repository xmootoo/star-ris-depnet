import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from logging import getLogger
import cvxpy as cp
import time
import os
from functorch import grad, vmap, hessian
from functools import partial
import math
logger = getLogger()


def get_W_Theta(config, w, phi_t, phi_r, *args):
    """
    works for both double and star RIS
    """
    if config.star:
        return get_W_Theta_star(config, w, phi_t, phi_r, args[0])
    else:
        return get_W_Theta_normal(config, w, phi_t, phi_r)

def get_W_Theta_normal(config, w, phi_t, phi_r):
    """
    double notmal RIS
    """
    amp, b = torch.split(w, [1, config.M*config.K*2], dim = 1)
    amp = torch.sigmoid(amp)
    b = b.reshape(-1, config.M, config.K, 2)/torch.norm(b ,2, dim = 1).reshape(b.shape[0], 1, 1, 1)
    W = amp.reshape(amp.shape[0], 1, 1)*(b[:,:,:,0] + 1j*b[:,:,:,1])*math.sqrt(config.P_max)
    Theta_t = torch.diag_embed(torch.exp(1j*phi_t))
    Theta_r = torch.diag_embed(torch.exp(1j*phi_r))
    return W, Theta_t, Theta_r

def get_W_Theta_star(config, w, phi_t, phi_r, a):
    """
    w: (B, 2KM + 1)   phi_t: (B, N) (output of linear layer)   a: (B, N) (output of linear layer)
    phi_r: (B, N) (output of linear layer)
    """
    amp, b = torch.split(w, [1, config.M*config.K*2], dim = 1)
    amp = torch.sigmoid(amp)
    b = b.reshape(-1, config.M, config.K, 2)/torch.norm(b ,2, dim = 1).reshape(b.shape[0], 1, 1, 1)
    W = amp.reshape(amp.shape[0], 1, 1)*(b[:,:,:,0] + 1j*b[:,:,:,1])*math.sqrt(config.P_max)

    coef = torch.sigmoid(a)
    a_t = (1 - config.utol)*coef + (1-coef)*(config.ltol)
    a_r = 1 - a_t

    Theta_t = torch.diag_embed(torch.sqrt(a_t)*torch.exp(1j*phi_t))
    Theta_r = torch.diag_embed(torch.sqrt(a_r)*torch.exp(1j*phi_r))
    return W, Theta_t, Theta_r


def get_inits(config, Data, seed):
    if config.star:
        return get_inits_star(config, Data, seed)
    else:
        return get_inits_normal(config, Data, seed)

def get_inits_star(config, Data, seed):
    # torch.manual_seed(seed = seed)
    H = Data['H']
    G = Data['G']
    H_t = H[:,:,0:2]
    H_r = H[:,:,2:]

    phi_t = torch.ones(H.shape[0], config.N).double()
    phi_r = torch.ones(H.shape[0], config.N).double()
    a = torch.ones(H.shape[0], config.N).double()
    w = torch.ones(H.shape[0], 2*config.M*config.K + 1).double()
    W, Theta_t, Theta_r = get_W_Theta(config, w, phi_t, phi_r, a)


    w_t = torch.bmm(H_t.conj().transpose(1, 2), torch.bmm(Theta_t, G))
    w_r = torch.bmm(H_r.conj().transpose(1, 2), torch.bmm(Theta_r, G))

    W = torch.cat([w_t, w_r], dim = 1).conj().transpose(1,2)
    W = (W/torch.norm(W, dim = (1,2)).reshape(W.shape[0], 1, 1)).unsqueeze(-1)
    w = torch.zeros(W.shape[0],  config.M*config.K*2 + 1).double()
    w[:,1:] = torch.cat([W.real, W.imag], dim = 3).reshape(W.shape[0], -1)
    w[:,0] = 10
    return w, phi_t, phi_r, a

def get_inits_normal(config, Data, seed):
    # torch.manual_seed(seed = seed)
    H = Data['H']
    G = Data['G']
    H_t = H[:,:config.N//2,0:2]
    H_r = H[:,config.N//2:,2:]
    G_t = G[:,:config.N//2, :]
    G_r = G[:,config.N//2:, :]

    phi_t = torch.ones(H.shape[0], config.N//2).double()
    phi_r = torch.ones(H.shape[0], config.N//2).double()
    w = torch.ones(H.shape[0], 2*config.M*config.K + 1).double()
    W, Theta_t, Theta_r = get_W_Theta(config, w, phi_t, phi_r)


    w_t = torch.bmm(H_t.conj().transpose(1, 2), torch.bmm(Theta_t, G_t))
    w_r = torch.bmm(H_r.conj().transpose(1, 2), torch.bmm(Theta_r, G_r))

    W = torch.cat([w_t, w_r], dim = 1).conj().transpose(1,2)
    W = (W/torch.norm(W, dim = (1,2)).reshape(W.shape[0], 1, 1)).unsqueeze(-1)
    w = torch.zeros(W.shape[0],  config.M*config.K*2 + 1).double()
    w[:,1:] = torch.cat([W.real, W.imag], dim = 3).reshape(W.shape[0], -1)
    w[:,0] = 10
    return w, phi_t, phi_r

def rate_vio(config, Data, w, phi_t, phi_r, *args):
    W, Theta_t, Theta_r = get_W_Theta(config, w, phi_t, phi_r, *args)
    R = rate(config, Data, W, Theta_t, Theta_r, users = True)
    diff = F.relu(config.r_min - R)
    return torch.linalg.norm(diff, ord = 2, dim = 1)**2
def V(config, Data, w, phi_t, phi_r, *args):
    return rate_vio(config, Data,  w, phi_t, phi_r, *args).sum()



def GD(config, Data, params, verbose = False, **kwargs):

    grad_w = grad(V, argnums=2)
    grad_phi_t = grad(V, argnums = 3)
    grad_phi_r = grad(V, argnums = 4)
    if config.star:
        grad_a = grad(V, argnums = 5)

    old_step_w = 0
    old_step_phi_r = 0
    old_step_phi_t = 0
    if config.star:
        old_step_a = 0
    args = []
    if 'init' in kwargs.keys():
        w = kwargs['init']['w']
        phi_t = kwargs['init']['phi_t']
        phi_r = kwargs['init']['phi_r']

        if config.star:
            a = kwargs['init']['a']
            args = [a]
    else:
        if config.star:
            w, phi_t, phi_r, a = get_inits(config, Data, seed = config.env_seed)
            args = [a]
        else:
            w, phi_t, phi_r = get_inits(config, Data, seed = config.env_seed)
    l1, l2 = 0,0
    stats = []
    stats.append(V(config, Data, w, phi_t, phi_r, *args).item())
    for i in range(params['num_iter']):
        l1 = V(config, Data, w, phi_t, phi_r, *args)
        gr_w = grad_w(config, Data, w, phi_t, phi_r, *args)
        gr_phi_t = grad_phi_t(config, Data, w, phi_t, phi_r, *args)
        gr_phi_r = grad_phi_r(config, Data, w, phi_t, phi_r, *args)
        if config.star:
            gr_a = grad_a(config, Data, w, phi_t, phi_r, a)

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

        l2 = V(config, Data, w, phi_t, phi_r, *args)
        stats.append(l2.item())
        if i%50 == 0:
            if verbose:
                print(l2.item())
            params['lr_w'] = 0.99*params['lr_w']
            params['lr_phi_r'] = 0.99*params['lr_phi_r']
            params['lr_phi_t'] = 0.99*params['lr_phi_t']
            if config.star:
                params['lr_a'] = 0.99*params['lr_a']
        if l2.item() == 0:
            if verbose:
                print(f"Converged --- loss value: {l2.item()}")
            break
    if config.star:
        return w, phi_t, phi_r, a, np.array(stats)
    else:
        return w, phi_t, phi_r, np.array(stats)

def nor_vecorize(config, Data, normalize = True):
    d = torch.zeros(Data['H'].shape[0], 2*config.N*(config.M + config.K)).double()
    b_size = Data['H'].shape[0]
    i = 0
    for k in ['G', 'H']:
        if normalize:
            real = ((Data[k].real - config.mean[k].real)/(config.std[k].real)).reshape(b_size, -1)
            img = ((Data[k].imag - config.mean[k].imag)/(config.std[k].imag)).reshape(b_size, -1)
        else:
            real = (Data[k].real).reshape(b_size, -1)
            img = (Data[k].imag).reshape(b_size, -1)
        d[:,i: i + real.shape[1]] = real
        i += real.shape[1]
        d[:,i: i + img.shape[1]] = img
        i += img.shape[1]
    return d

def get_features(config, Data, normalize = True):
    H = Data['H']
    G = Data['G']
    temp = torch.einsum('bkn,bnm->bnkm', H.transpose(1,2), G)
    F = torch.cat([temp.real, temp.imag], dim = 2)
    if normalize:
        F = (F - config.f_mean)/(config.f_std)
    F = torch.einsum('bnkm->bkmn', F)
    return F

def get_feature_stat(config):
    mean_r = torch.einsum('kn,nm->nkm', config.mean['H'].real.transpose(0,1), config.mean['G'].real)
    mean_img = torch.einsum('kn,nm->nkm', config.mean['H'].imag.transpose(0,1), config.mean['G'].imag)
    phi_h_real, phi_h_img = (config.std['H'].real)**2, (config.std['H'].imag)**2
    phi_g_real, phi_g_img = (config.std['G'].real)**2, (config.std['G'].imag)**2

    def get_var(config, phi_1, phi_2, mean_1, mean_2):
        temp_1 = torch.einsum('kn,nm->nkm', phi_1, phi_2)
        temp_2 = torch.einsum('kn,nm->nkm', phi_1, mean_2**2)
        temp_3 = torch.einsum('kn,nm->nkm', mean_1**2, phi_2)
        return temp_1 + temp_2 + temp_2
    phi_r = get_var(config, phi_h_real.transpose(0,1), phi_g_real, config.mean['H'].real.transpose(0,1), config.mean['G'].real)
    phi_img = get_var(config, phi_h_img.transpose(0,1), phi_g_img, config.mean['H'].imag.transpose(0,1), config.mean['G'].imag)

    mean = torch.cat([mean_r, mean_img], dim = 1).unsqueeze(0)
    std = torch.sqrt(torch.cat([phi_r, phi_img], dim = 1)).unsqueeze(0)
    config.f_mean = mean
    config.f_std = std

def rate(config, Data, W, Theta_t, Theta_r, users = False):
    if config.star:
        return rate_star(config, Data, W, Theta_t, Theta_r, users)
    else:
        return rate_normal(config, Data, W, Theta_t, Theta_r, users)

def rate_star(config, Data, W, Theta_t, Theta_r, users = False):
    """
    data: ('G', 'H')
    'G': (B, N, M)    'H': (B, N, K)   'W': (B, M, K)   'Theta': (B, N, N)
    """
    check = list(set([Theta_t.shape[1], Theta_t.shape[2], Theta_r.shape[1], Theta_r.shape[2]]))
    assert  len(check) == 1 and check[0] == config.N, 'Wrong Configurations for Star IRS!'
    H = Data['H']
    G = Data['G']
    H_t = H[:,:,0:2]
    H_r = H[:,:,2:]
    signals_t = torch.bmm(torch.bmm(H_t.conj().transpose(1, 2), torch.bmm(Theta_t, G)), W)
    signals_r = torch.bmm(torch.bmm(H_r.conj().transpose(1, 2), torch.bmm(Theta_r, G)), W)
    all_signals = torch.cat([signals_t, signals_r], dim = 1)
    all_signals = all_signals.abs()**2

    recieved_mask = torch.eye(config.K).unsqueeze(0).to(config.device)
    infer_mask = torch.ones(config.K).unsqueeze(0).to(config.device) - recieved_mask
    rec_signal = (recieved_mask*all_signals).sum(-1)
    infer_signal = (infer_mask*all_signals).sum(-1)
    sinr = rec_signal/(infer_signal + config.N0)
    rate = torch.log(1 + sinr)/torch.log(torch.tensor(2))

    if users:
        sum_rate = rate
    else:
        sum_rate = rate.sum(-1)
    return sum_rate

def rate_normal(config, Data, W, Theta_t, Theta_r, users = False):
    """
    data: ('G', 'H')
    'G': (B, N, M)    'H': (B, N, K)   'W': (B, M, K)   'Theta': (B, N, N)
    """
    check = list(set([Theta_t.shape[1], Theta_t.shape[2], Theta_r.shape[1], Theta_r.shape[2]]))
    assert  len(check) == 1 and check[0] == config.N//2, 'Wrong Configurations for Normal IRS!'
    H = Data['H']
    G = Data['G']
    H_t = H[:,:config.N//2,0:2]
    H_r = H[:,config.N//2:,2:]
    G_t = G[:,:config.N//2, :]
    G_r = G[:,config.N//2:, :]
    signals_t = torch.bmm(torch.bmm(H_t.conj().transpose(1, 2), torch.bmm(Theta_t, G_t)), W)
    signals_r = torch.bmm(torch.bmm(H_r.conj().transpose(1, 2), torch.bmm(Theta_r, G_r)), W)
    all_signals = torch.cat([signals_t, signals_r], dim = 1)
    all_signals = all_signals.abs()**2

    recieved_mask = torch.eye(config.K).unsqueeze(0).to(config.device)
    infer_mask = torch.ones(config.K).unsqueeze(0).to(config.device) - recieved_mask
    rec_signal = (recieved_mask*all_signals).sum(-1)
    infer_signal = (infer_mask*all_signals).sum(-1)
    sinr = rec_signal/(infer_signal + config.N0)
    rate = torch.log(1 + sinr)/torch.log(torch.tensor(2))

    if users:
        sum_rate = rate
    else:
        sum_rate = rate.sum(-1)
    return sum_rate
