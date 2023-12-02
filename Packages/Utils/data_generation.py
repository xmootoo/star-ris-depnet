import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import itertools
from .comlib import *
from tqdm import tqdm
import torch
import pickle
from functools import partial

def generate_path_loss(config, rng, num_samples, fix_users = True):
    """
    Generates path_loss datasets
    G: (B, N, M)
    H: (B, N, K)
    B: num_samples    N: num of RIS elements   M: num of BS antennas  K: num of users
    works for normal and star RIS.
    """
    PL = lambda x: np.array(35.6 + 22*np.log10(x))
    if fix_users:
        rng_fix = np.random.default_rng(seed = 123)
        v = np.repeat(rng_fix.uniform(size = (1, 1, 1))*2*np.pi, num_samples, axis = 0)
        phi = np.repeat(rng_fix.uniform(size = (1, 1, 1))*2*np.pi, num_samples, axis = 0)
        si = np.repeat(rng_fix.uniform(size = (1, 1, config.K))*2*np.pi, num_samples, axis = 0)
    else:
        v = rng.uniform(size = (num_samples, 1, 1))*2*np.pi
        phi = rng.uniform(size = (num_samples, 1, 1))*2*np.pi
        si = rng.uniform(size = (num_samples, 1, config.K))*2*np.pi

    D_n = np.arange(config.N).reshape(1, config.N, 1)
    a_g_N = np.exp(1j*(np.sin(v)*D_n)*np.pi)
    D_m = np.arange(config.M).reshape(1, config.M, 1)
    a_g_M = np.exp(1j*(np.sin(phi)*D_m)*np.pi)
    a_h_N = np.exp(1j*(np.sin(si)*D_n)*np.pi)

    G_bar  = (rng.normal(size = (num_samples, config.N, config.M)) + 1j*rng.normal(size = (num_samples, config.N, config.M)))
    H_bar = (rng.normal(size = (num_samples, config.N, config.K)) + 1j*rng.normal(size = (num_samples, config.N, config.K)))
    if fix_users:
        r = np.repeat(rng_fix.uniform(size = (1, config.K, 1))*10, num_samples, axis = 0)
        theta = np.repeat(rng_fix.uniform(size = (1, config.K, 1))*2*np.pi, num_samples, axis =0)
    else:
        r = rng.uniform(size = (num_samples, config.K, 1))*10
        theta = rng.uniform(size = (num_samples, config.K, 1))*2*np.pi
    d = np.sqrt(np.power(r,2) + np.power(30,2) -2*r*30*np.cos(theta))

    L1 = np.power(10, (-PL(200)/10).reshape(1,1,1))
    L2 = np.einsum('bki->bik', np.power(10, (-PL(d)/10)))

    G_los = np.einsum('bni,bmi->bnm',a_g_N, a_g_M.conj())
    # L1 = 1
    # L2 = 1
    G = np.sqrt(L1)*(np.sqrt(config.epsilon/(config.epsilon + 1))*G_los + np.sqrt(1/(config.epsilon + 1))*G_bar)
    H = np.sqrt(L2)*(np.sqrt(config.epsilon/(config.epsilon + 1))*a_h_N + np.sqrt(1/(config.epsilon + 1))*H_bar)
    return G, H

def generate_feasible_data(config, rng, num_data, batch_size):
    main_pbar = tqdm(total = num_data, desc= "Number of generated Feasible datapoints", position = 0)
    num_added = 0
    G_all = np.zeros((num_data, config.N, config.M)) + 1j*np.zeros((num_data, config.N, config.M))
    H_all = np.zeros((num_data, config.N, config.K)) + 1j*np.zeros((num_data, config.N, config.K))
    gd_params = {}
    while num_added < num_data:
#         print(f'num added: {num_added}')
        for k, v in [('lr_w', 1), ('lr_phi_t', 1), ('lr_phi_r', 1), ('lr_a', 1), ('momentum_phi_t', 0.5), ('momentum_phi_r', 0.5), ('momentum_w', 0.5), ('momentum_a', 0.5), ('num_iter', 100)]:
            gd_params[k] = v
        g, h = generate_path_loss(config, rng, batch_size, True)
        data = {'G': torch.from_numpy(g), 'H': torch.from_numpy(h)}
        # torch.manual_seed(seed = 10)
        # phi_t = torch.rand(batch_size, config.N).double()*2*torch.pi
        # w = torch.rand(batch_size, config.M*config.K*2 + 1).double()
        # a = torch.rand(batch_size, config.N).double()
        # , init = {'w': w, 'phi_t': phi_t, 'a': a}
        idx = torch.ones(data['H'].shape[0])
        for flag in [True, False]:
            config.star = flag
            args = []
            if flag:
                w_hat, phi_t_hat, phi_r_hat, a_hat, stat = GD(config, data, gd_params, verbose = False)
                args.append(a_hat)
            else:
                w_hat, phi_t_hat, phi_r_hat, stat = GD(config, data, gd_params, verbose = False)
            temp = (rate_vio(config, data, w_hat, phi_t_hat, phi_r_hat, *args) <= 1e-30)
            idx = torch.logical_and(idx, temp)
        add = idx.sum().item()
        if (num_added + add) <= num_data:
            G_all[num_added: num_added + add, :, :] = g[idx,:,:]
            H_all[num_added: num_added + add, :, :] = h[idx,:,:]
            num_added += add
        else:
            G_all[num_added:, :, :] = g[idx,:,:][:num_data - num_added,:,:]
            H_all[num_added:, :, :] = h[idx,:,:][:num_data - num_added,:,:]
            num_added = num_data
        main_pbar.update(add)
    main_pbar.close()
    return G_all, H_all
