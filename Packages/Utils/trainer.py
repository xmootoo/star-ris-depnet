import os
import time
from logging import getLogger
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from .datahandler import create_data_loader
import torch.optim as optim
from .utils import to_cuda
from .objective import *
from .comlib import *
logger = getLogger()

class Trainer(object):
    """
    This class is responsible for training the model
    """
    def __init__(self, config, model, **kwargs):
        """
        Initialize trainer.
        """
        # modules / params
        self.config = config
        self.model = model
        self.model_type = config.model_type
        # epoch / iteration size
        self.epoch_size = config.epoch_size
        self.batch_size = config.batch_size
        # Optimizer
        self.optimizer = self.set_optimizer()
        lambda1 = lambda epoch: self.config.decay_rate ** epoch
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda1)
        # loss function
        self.loss = loss_rp if self.config.model_type.lower() == 'wtrpnet' else loss_n
        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.proj_on = config.proj_on
        self.N0 = config.N0
        self.n_samples = 0
        self.stats = OrderedDict(
            [('processed_data_samples', 0), ('loss', []), ('rate', []), ('vio', [])]
        )
        self.batch_stats = {'loss': defaultdict(lambda: []), 'rate': defaultdict(lambda: []), 'vio': defaultdict(lambda: [])}
        self.epoch_stats = {'loss': defaultdict(np.float64), 'rate': defaultdict(np.float64), 'vio': defaultdict(np.float64)}
        self.last_time = time.time()
        # metrics
        self.stopping_criterion = None
        # reload potential checkpoints
        self.check_path = os.path.join(self.config.exp_dir, f'Model{self.config.model_id}')
        self.check_name = self.config.check_name
        if not os.path.exists(self.check_path):
            os.makedirs(self.check_path)
        self.reload_checkpoint()
        # create data loaders
        if not config.eval_only:
            dtype = 'train'
            self.data = create_data_loader(config, dtype = dtype)
            self.epoch_size = len(self.data.dataset) if config.epoch_size == -1 else config.epoch_size
            self.dataloader = iter(self.data)


    def set_optimizer(self):
        """
        Set the optimizer
        """
        optim_type = self.config.optimizer
        if optim_type == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr = self.config.learning_rate, weight_decay = self.config.weight_decay)
        elif optim_type == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr = self.config.learning_rate)
        return optimizer

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")

        self.optimizer.zero_grad()
        loss.backward()
        if sum(torch.sum(torch.isnan(p.grad)) for p in self.model.parameters() if p.requires_grad) == 0:
            if self.config.clip_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), max_norm = self.config.clip_grad_norm)

            self.optimizer.step()
        else:
            logger.warning("NaN detected in model gradients!!!")

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 20 != 0:
            return
        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k.upper().replace('_', '-'), np.sum(v)/self.stats['processed_data_samples'])
            for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = ""
        s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.4e}".format(group['lr']) for group in self.optimizer.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:8.2f} samples/s - ".format(
            self.stats['processed_data_samples'] * 1.0 / diff
        )
        self.stats['processed_data_samples'] = 0
        self.stats['loss'] = []
        self.stats['rate'] = []
        self.stats['vio'] = []
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """

        path = os.path.join(self.check_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            'epoch': self.epoch,
            'stats': {k: v for k,v in self.stats.items()},
            'n_total_iter': self.n_total_iter,
            'config': {k: v for k, v in self.config.__dict__.items()},
        }
        for ke in self.epoch_stats.keys():
            data[f'epoch_stats_{ke}'] = {k: v for k,v in self.epoch_stats[ke].items()}
            data[f'batch_stats_{ke}'] = {k: v for k,v in self.batch_stats[ke].items()}
        logger.warning("Saving Model parameters ...")
        data['model'] = self.model.state_dict()

        if include_optimizers:
            logger.warning("Saving Model optimizer ...")
            data['model_optimizer'] = self.optimizer.state_dict()

        torch.save(data, path)
        if self.config.model_type.lower() == 'iternn':
            self.model.save(self.model_main_dir, f'{name}.pth')
    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.check_path, self.check_name)
        if not os.path.isfile(checkpoint_path):
            if self.config.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.config.reload_checkpoint
                assert os.path.isfile(checkpoint_path), 'Could not find the specified checkpoint'

        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location='cpu')

        # reload model parameters
        self.model.load_state_dict(data['model'])

        # reload optimizers
        logger.warning("Reloading checkpoint optimizer model ...")
        self.optimizer.load_state_dict(data['model_optimizer'])
        for g in self.optimizer.param_groups:
            g['lr'] = self.config.learning_rate
            g['weight_decay'] = self.config.weight_decay
        # reload main metrics
        self.epoch = data['epoch'] + 1
        for item in ['stats']:
            for k in data[item]:
                temp = getattr(self, f'{item}')
                temp[k] = data[item][k]
        for ke in self.epoch_stats.keys():
            for k,v in data[f'epoch_stats_{ke}'].items():
                self.epoch_stats[ke][k] = v
            for k,v in data[f'batch_stats_{ke}'].items():
                self.batch_stats[ke][k] = v
        self.n_total_iter = data['n_total_iter']
        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if self.config.save_periodic > 0 and self.epoch % self.config.save_periodic == 0:
            self.save_checkpoint('periodic-%i' % self.epoch)

    def end_epoch(self):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None :
            # not implemented yet
            pass
        for k in self.epoch_stats.keys():
            self.epoch_stats[k][f'epoch-{self.epoch}'] = np.sum(self.batch_stats[k][f'epoch-{self.epoch}'])/self.n_samples
        message = "trianing stats: "
        for k in self.epoch_stats.keys():
            message += f"{k} -> {self.epoch_stats[k][f'epoch-{self.epoch}']: .4f}  "
        self.config.train_params['num_iter'] += self.config.train_increase
        message += f"train projection iteration updated to {self.config.train_params['num_iter']}"
        logger.info(message)
        self.save_checkpoint('checkpoint')
        self.epoch += 1
        self.dataloader = iter(self.data)
        self.scheduler.step()

    def step(self):
        """
        Perform one step over one batch of data samples.
        """
        Model = self.model
        Model.train()
        data = next(self.dataloader)
        h, g = to_cuda(self.config, data['H'], data['G'])
        data = {}
        data['H'] = h
        data['G'] = g
        if hasattr(Model, 'project'):
            if self.config.project:
                if self.n_iter % self.proj_on == 0:
                    Model.project = True
                else:
                    Model.project = False
            else:
                Model.project = False
        if self.config.model_type.lower() == 'wtrpnet':
            Model.adjust(self.config.active_dnns)
            w, phi_t, phi_r, r_min, p_max = Model(data, self.config.init)
            W, Theta_t, Theta_r = get_W_Theta(self.config, w, phi_t, phi_r, mode1 = self.config.mode1)
            loss = self.loss(self.config, data, phi_t, phi_r, w, r_min, p_max)
        elif self.config.model_type.lower() == 'wtnet':
            args = []
            if self.config.star:
                w, phi_t, phi_r, a = Model(data)
                args = [a]
            else:
                w, phi_t, phi_r = Model(data)
            W, Theta_t, Theta_r = get_W_Theta(self.config, w, phi_t, phi_r, *args)
            loss = self.loss(self.config, data, W, Theta_t, Theta_r)

        r = rate(self.config, data, W, Theta_t, Theta_r).mean()
        vio = rate_vio(self.config, data, w, phi_t, phi_r, *args).mean()
        if (loss != loss).data.any():
            l = torch.zeros_like(loss).to(self.config.device)
            logger.warning("NaN detected")
        else:
            l = loss
        for k, v in [('loss', l), ('rate', r), ('vio', vio)]:
            self.stats[k].append(v.item()*data['H'].shape[0])
            self.batch_stats[k][f'epoch-{self.epoch}'].append(v.item()*data['H'].shape[0])
            # optimize
        self.optimize(loss)

            # number of processed data samples
        self.stats['processed_data_samples'] += data['H'].shape[0]
        self.n_samples += data['H'].shape[0]
