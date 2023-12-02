"""
This the main module for the deep learning tutorial. The hardware setup is based on single-GPU.

"""
from Packages import *
from Packages import Utils, Models

# from tqdm import trange
import numpy as np
import argparse
import random
import torch
import os
import json
from shutil import copyfile
## Raise an Error for every floating-point operation error: devition by zero, overfelow, underflow, and invalid operation
np.seterr(all='raise')


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Experiment Configurations")

    # main parameters
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=int, default=0,
                        help="Experiment ID (if 0, generate a new one)")
    parser.add_argument("--env_seed", type=int, default=0,
                        help="Base seed for environments (-1 to use timestamp seed)")
    parser.add_argument("--device", type=str, default='cpu',
                        help="Specifies the device (default: 'cpu')")

    # Dataset parameters

    parser.add_argument("--data_preparation", type=bool_flag, default=False,
                        help="Preprocess the data before training")
    parser.add_argument("--problem", type=str, default='U',
                        help="Specifies the problem at hand {P or U} (default: 'U')")
    parser.add_argument("--dataset_id", type=str, default='1',
                        help="Specifies the id of the dataset (default: '1')")
    parser.add_argument("--legacy_datasets", type=bool_flag, default=False,
                        help="The main directory of the dataset")
    parser.add_argument("--reyleigh", type=bool_flag, default=False,
                        help="The channel model of the dataset")
    parser.add_argument("--qos", type=float, default=0.75,
                        help="Quality of Service")

    ## Data prepration for Debug
    parser.add_argument("--debug", type=bool_flag, default=False,
                        help="Change to debug mode")
    parser.add_argument("--debug_id", type=int, default=0,
                        help="Specifies the id of the debug dataset")
    parser.add_argument("--debug_train_samples", type=int, default=1,
                        help="Number of datapoints in train/data.mat")
    parser.add_argument("--debug_test_samples", type=int, default=1,
                        help="Number of datapoints in test/data.mat")
    parser.add_argument("--debug_common_samples", type=int, default=1,
                        help="Number of common datapoints between train and test set")
    parser.add_argument("--debug_seed", type=int, default=1,
                        help="The seed of debug dataset (-1 to use environment seed)")
    ## Whether to normalize or augment the dataset beforehand
    parser.add_argument("--normalize", type=bool_flag, default=True,
                        help="Standardizes the dataset (Default: False)")
    parser.add_argument("--augment", type=bool_flag, default=False,
                        help="Augments data points in each batch")
    parser.add_argument("--pre_augment", type=bool_flag, default=False,
                        help="Augments all data points before making the batches")
    # model parameters
    parser.add_argument("--model_type", type=str, default="PNet_wor",
                        help="Set the model type {JUPNet}")

    parser.add_argument("--A_mode", type=str, default="A_init",
                        help="Set the Association source {A_init, A_model}")
    parser.add_argument("--P_mode", type=str, default="P_init",
                        help="Set the Power source {P_init, P_model}")

    # training parameters

    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of samples per batch")
    parser.add_argument("--epoch_size", type=int, default=10000,
                        help="Number of samples per epoch (-1 for everything)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer")
    parser.add_argument("--decay_rate", type=float, default=0.99,
                        help="Learning rate decay for scheduler")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="L_2 weight decay")
    parser.add_argument("--clip_grad_norm", type=float, default=0,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--max_epoch", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default='Adam',
                        help="The type of optimization algorithm (default Adam)")

    # reload data
    parser.add_argument("--train_reload_size", type=int, default=-1,
                        help="Reloaded training set size (-1 for everything)")
    parser.add_argument("--test_reload_size", type=int, default=-1,
                        help="Reloaded training set size (-1 for everything)")

    # reload pretrained model / checkpoint
    parser.add_argument("--load_model", type=bool_flag, default=False,
                        help="Load a pretrained model")
    parser.add_argument("--model_path", type=str, default= '',
                        help="Path of the pretrained model (default 'trained/model.pth')")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")
    # Experiment setup
    parser.add_argument("--num_user", type=int, default=4,
                        help="Number of users")
    parser.add_argument("--quota", type=int, default=2,
                        help="Quota of each BS")
    parser.add_argument("--num_bs", type=int, default=2,
                        help="Number of base stations")

    # evaluation
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    # parser.add_argument("--eval_every", type=int, default=1,
    #                     help="Evaluate on validation and test sets every # epochs")
    return parser

def save(config, data):
    """
    Saves the scores of all the evaluations in the experiment directory
    """
    path = os.path.join(config.exp_dir, 'scores.pickle')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            old_data = pickle.load(f)

        for key in old_data.keys():
            old_data[key].update(list(zip(data[key].keys(), data[key].values())))
        data = old_data
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def main(config):
    data = {}
    # initialize the logger
    # log the current experiment setup
    logger = config.get_logger()
    set_seed(config)
    # Data prepration
    if config.data_preparation:
        logger.info('============ Starting the data preprocessing ============')
        if config.debug:
            logger.info('============ Starting to make debug dataset ============')
            make_debug_dataset(config)
            logger.info('============ Making debug dataset finished ============')
        logger.info('============ End of data preprocessing ============')
        copyfile(os.path.join(config.exp_dir, 'train.log'), os.path.join(config.Dataset_dir, 'Info.txt'))
        return
    # Create model
    model = build_model(config)
    # Creates trainer
    trainer = Trainer(config, model)
    # Create evaluator
    evaluator = Evaluator(trainer)

    visualizer = Visualizer(config, trainer, evaluator)
    # visualizer.draw_model()
    visualizer.weight()
    if not config.eval_only:
        for epoch in range(config.max_epoch):
             logger.info("============ Starting epoch %i ... ============" % trainer.epoch)
             trainer.n_samples = 0
             torch.cuda.empty_cache()
             while trainer.n_samples < trainer.epoch_size:
                 # training steps
                 ## passing over one batch of data
                 trainer.step()
                 trainer.iter()
             logger.info("============ End of epoch %i ============" % trainer.epoch)
             # if epoch%config.eval_every == 0:
             scores = evaluator.run_all_evals()
             # print / JSON log
             for k, v in scores.items():
                 logger.info("%s -> %.6f" % (k, v))
             logger.info("__log__:%s" % json.dumps(scores))
             # if trainer.config.P_mode.lower() == 'p_model':
             #     trainer.config.P_mode = 'P_init'
             # else:
             #     trainer.config.P_mode = 'P_model'
             # end of epoch
             trainer.save_periodic()
             trainer.end_epoch()
             data = {}
             data['evaluation_score'] = evaluator.scores
             data['best_criterion'] = evaluator.best_criterion
             for ke in trainer.epoch_stats.keys():
                 data[f'epoch_stats_{ke}'] = {k: v for k,v in trainer.epoch_stats[ke].items()}
                 data[f'batch_stats_{ke}'] = {k: v for k,v in trainer.batch_stats[ke].items()}
             save(config, data)
             visualizer.record()
    else :
        scores = evaluator.run_all_evals()
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
    visualizer.close()

if __name__ == '__main__':
    # Parse input
    args = get_parser()
    args = args.parse_args()

    config = Config(args)
    # Call main
    main(config)
