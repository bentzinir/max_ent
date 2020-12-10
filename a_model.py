import torch
import torch.nn as nn
import torch.nn.functional as F


class AModelTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam()
        self.loss = F.cross_entropy()
