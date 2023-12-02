import os
import sys
import re
import argparse
import pickle
import torch
import numpy as np
from .logger import create_logger
from .datahandler import *

class Config():
    """
    This class is responsible for organizing the current experiment configurations across all the units
    """
    def __init__(self, arg):
        # Define dataset path
        self.__set_attr(arg)
        self.__sanity_check()
        self.get_data_dir()

        # Experiment path
        self.dump_path = os.path.join(os.getcwd(), 'Dumped')
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
        if arg.exp_id == 0 :
            self.exp_id = self.get_exp_id()
        else :
            self.exp_id = arg.exp_id
        self.exp_dir = os.path.join(self.dump_path, str(self.exp_id))
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        # Experimental setup
        if self.device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        #Training configurations
        self.model_path = os.path.join(os.getcwd(), 'trained', 'model.pth') if arg.model_path == '' else arg.model_path
        if self.load_model:
            assert os.path.exists(self.model_path)

        # wireless network settings
        self.N0 = 10**(-15)
        self.N = 64
        self.M = 8
        self.K = 4
        self.star = True
        self.num_bs = 1
        self.gamma = 0
        self.mean = 0
        self.std = 0
        self.utol = 1e-5
        self.ltol = 1e-5
        self.P_max = 10
        self.epsilon = 10
        self.r_min = 2
        self.train_params = {}
        self.test_params = {}
        for k, v in [('lr_w', 0.01), ('lr_phi_t', 0.01), ('lr_phi_r', 0.01), ('lr_a', 0.01), ('momentum_phi_t', 0.5), ('momentum_phi_r', 0.5), ('momentum_w', 0.5), ('momentum_a', 0.5), ('num_iter', 100)]:
            self.train_params[k] = v
            self.test_params[k] = v
        df_test = create_data_loader(self, 'test')
        data = next(iter(df_test))
        self.save()

    def __set_attr(self, arg):
        """
        For each element in argument parser, defines a corresponding attribute
        for config
        """
        for  k, v in vars(arg).items():
            setattr(self, k, v)
        assert  (not self.data_preparation) or self.debug , 'Data preprocessing only works for debug mode!!!'
        self.eval_dtypes = ['test', 'valid']
    def __sanity_check(self):
        """
        Checks if the current input arguments are reasonable
        """
        if self.data_preparation:
            if self.debug:
                train = self.debug_train_samples - self.debug_common_samples
                test = self.debug_test_samples - self.debug_common_samples
                assert (train > -1) and (test > -1) , 'Number of common datapoints cannot be greater than the set size'

    def get_exp_id(self):
        """
        Returns an integer as an experiment id.
        """
        dir_list = os.listdir(self.dump_path)
        if len(dir_list) == 0 :
            id = 1
        else :
            dir_list = [int(dir) for dir in dir_list]
            id = max(dir_list) + 1
        return id

    def get_logger(self):
        """
        creates a logger
        """
        command = ["python", sys.argv[0]]
        for x in sys.argv[1:]:
            if x.startswith('--'):
                assert '"' not in x and "'" not in x
                command.append(x)
            else:
                assert "'" not in x
                if re.match('^[a-zA-Z0-9_]+$', x):
                    command.append("%s" % x)
                else:
                    command.append("'%s'" % x)
        command = ' '.join(command)
        self.command = command + ' --exp_id "%s"' % self.exp_id

        # create a logger
        logger = create_logger(os.path.join(self.exp_dir, 'train.log'))
        logger.info("============ Initialized logger ============")
        logger.info("\n".join("%s: %s" % (k, str(v))
                              for k, v in sorted(dict(vars(self)).items()) if k not in ['mean', 'std']))
        logger.info("The experiment will be stored in %s\n" % self.exp_dir)
        logger.info("Running command: %s" % command)
        logger.info("")
        return logger

    def get_data_dir(self):
        """
        Locates the location of the dataset
        """
        main_dir = os.path.join('Star')

        if self.data_preparation:
            raise NotImplementedError

        else:
            if self.debug:
                raise NotImplementedError
            else:
                sub_dir = 'Primary'
            self.Dataset_dir = os.path.join(os.getcwd(), 'Datasets', main_dir, self.dataset_id)
            assert os.path.exists(self.Dataset_dir), 'Dataset directory does not exists!!!'
            self.train_dir = os.path.join(self.Dataset_dir, 'train')
            self.test_dir = os.path.join(self.Dataset_dir, 'test')
            self.valid_dir = os.path.join(self.Dataset_dir, 'valid')
            assert all([os.path.exists(dir) for dir in [self.test_dir, self.train_dir, self.valid_dir ]]), 'train, test, valid dir is not there!!!'
            if self.normalize:
                self.norm_data = os.path.join(os.getcwd(), 'Datasets', main_dir, self.dataset_id, 'train')

    def update(self):
        """
        Updates some cofigurations in runtime based on the changes made on the other relevant configurations
        """
        self.__sanity_check()
        self.get_data_dir()

    def save(self):
        """
        Saves the config object
        """
        path = os.path.join(self.exp_dir, 'config.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
