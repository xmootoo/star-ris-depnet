from logging import getLogger
from collections import OrderedDict, defaultdict
from .datahandler import create_data_loader
import os
import torch
from .utils import to_cuda
from sklearn.metrics import accuracy_score
import numpy as np
from .objective import *
import torch.nn.functional as F
from .comlib import *
import pickle
logger = getLogger()

class Evaluator(object):
    """
    This class is responsible for evaluating the performance of the model based on the validation metrics

    """
    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.model = trainer.model
        self.config = trainer.config
        self.scores = OrderedDict()
        self.best_criterion = {'test':0,'train':0,'valid':0}
        self.dtypes = self.config.eval_dtypes
        self.model_type = self.config.model_type
        self.update('scores.pickle')
    def update(self, name):
        """
        Updates the stats of evaluator based on the previous round
        """
        path = os.path.join(self.config.exp_dir, name)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            keys = [('evaluation_score', 'scores'), ('best_criterion', 'best_criterion')]
            for pair in keys:
                old, new = pair
                temp = getattr(self, new)
                temp.update(list(zip(data[old].keys(), data[old].values())))
                setattr(self,new, temp)
            logger.warning(f"Evaluator Updated!!")
    def run_all_evals(self):
        """
        Run all evaluations.violation_probx
        """
        scores = OrderedDict({'epoch': self.trainer.epoch})
        with torch.no_grad():
            dtypes = self.dtypes
            for data_type in dtypes:
                for task in [self.config.problem]:
                    self.eval(data_type, task, scores)
        self.scores[f'epoch-{self.trainer.epoch}'] = scores
        return scores

    def eval(self, data_type, task, scores):
        """
        Running evaluation.
        """
        config = self.config
        Model = self.model
        Model.eval()
        assert task in ['U', 'P']
        # iterator
        self.iterator = create_data_loader(config=config, dtype = data_type)
        # self.eval_size = len(self.iterator.dataset)
        self.eval_size = 0
        score = defaultdict(lambda : torch.tensor(0, device = self.config.device).double())
        self.get_score(Model, score)

        # compute perplexity and prediction accuracy
        for key in score.keys():
            scores[f'{data_type}_{task}_{key}'] = float(score[key]/self.eval_size)
        key = 'NN_sum_rate'
        if scores[f'{data_type}_{task}_{key}'] > self.best_criterion[data_type]:
            self.best_criterion[data_type] = scores[f'{data_type}_{task}_{key}']
            self.trainer.save_checkpoint(f'best_model_{data_type}')

    def get_score(self, Model, score):
        """
        For every data point in dtype set, calculate the loss, sum rate, and constraint violation value

        """
        for data in self.iterator:
            h, g = to_cuda(self.config, data['H'], data['G'])
            data = {}
            data['H'] = h
            data['G'] = g
            if hasattr(Model, 'project'):
                Model.project = self.config.project

            if self.config.model_type.lower() == 'wtrpnet':
                w, phi_t, phi_r, r_min, p_max = Model(data, self.config.init)
                l = loss_rp(self.config, data, phi_t, phi_r, w, r_min, p_max)
                W, Theta_t, Theta_r = get_W_Theta(self.config, w, )
            elif self.config.model_type.lower() == 'wtnet':
                args = []
                if self.config.star:
                    w, phi_t, phi_r, a = Model(data)
                    args = [a]
                else:
                    w, phi_t, phi_r = Model(data)
                W, Theta_t, Theta_r = get_W_Theta(self.config, w, phi_t, phi_r, *args)
                l = loss_n(self.config, data, W, Theta_t, Theta_r)

            r = rate(self.config, data, W, Theta_t, Theta_r)
            vio = rate_vio(self.config, data, w, phi_t, phi_r, *args)
            score['loss'] += l*data['H'].shape[0]
            score['NN_sum_rate'] += r.sum()
            score['NN_QoS_penalty'] += vio.sum()
            score['NN_Violation_prob'] += (vio > 1e-6).sum()
            self.eval_size += data['H'].shape[0]
