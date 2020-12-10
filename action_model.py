import torch
import torch.nn as nn
import torch.nn.functional as F


class AModelTrainer:
    def __init__(self, model, lr, adam_epsilon=1e-7):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=adam_epsilon)
        self.loss = F.cross_entropy()

    def forward(self, s, s_prime):
        x = torch.cat((s, s_prime), -1)
        x = self.model(x)
        return x
