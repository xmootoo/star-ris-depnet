import os
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from .datahandler import create_data_loader
from .utils import to_cuda
import torch
logger = getLogger()

class Visualizer:
    def __init__(self, config, trainer, evaluator):
        self.path = os.path.join(config.exp_dir, 'run')
        self.trainer = trainer
        self.evaluator = evaluator
        self.config = config
        self.writer = SummaryWriter(self.path)
        logger.info(f"============ Visualizer created-Save in {self.path} ============")
    def draw_model(self):
        None
        # model = self.trainer.model
        # data = next(iter(create_data_loader(config=self.config, dtype = 'test', load_from_file = True)))
        # data = to_cuda(self.config, *data)
        # self.writer.add_graph(model, (data))
    def record(self):
        epoch = self.trainer.epoch - 1
        score = self.evaluator.scores[f'epoch-{epoch}']
        model = self.trainer.model
        self.writer.add_scalar("Training loss", self.trainer.epoch_stats['loss'][f'epoch-{epoch}'], epoch)
        self.writer.add_scalar("Testing loss", score[f'test_{self.config.problem}_loss'], epoch)
        data = {k: v for k,v in score.items() if 'sum_rate' in k}
        data['train_sum_rate'] = self.trainer.epoch_stats['rate'][f'epoch-{epoch}']
        self.writer.add_scalars("Test sum rate", data, epoch)
        data = {k: v for k,v in score.items() if 'QoS_penalty' in k}
        data['train_QoS_penalty'] = self.trainer.epoch_stats['vio'][f'epoch-{epoch}']
        self.writer.add_scalars("Test QoS penalty", data, epoch)
        data = {k: v for k,v in score.items() if 'Violation_prob' in k}
        self.writer.add_scalars("Test Violation prob", data, epoch)
        for name, weight in model.named_parameters():
            self.writer.add_histogram(name,weight, epoch)
            if weight.requires_grad:
                self.writer.add_histogram(f'{name}.grad',weight.grad, epoch)
            # t = torch.clone(weight.grad/weight)
            # if t.isnan().any():
            #     t[t.isnan()] = 0
            # self.writer.add_histogram(f'{name}.grad/{name} weight',(t).abs(), epoch)
    def weight(self):
        epoch = -1
        model = self.trainer.model
        for name, weight in model.named_parameters():
            self.writer.add_histogram(name,weight, epoch)
            # self.writer.add_histogram(f'{name}.grad',weight.grad, epoch)
            # self.writer.add_histogram(f'{name}.grad/{name} weight',(weight.grad/weight).abs(), epoch)
    def close(self):
        self.writer.close()
