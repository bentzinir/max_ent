from stable_baselines3.common import logger
import torch
import numpy as np
from collections import deque
from ensemble.common.format_string import format_string
from stable_baselines3.common.utils import polyak_update


class DiscriminatorTrainer:
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, discrimination_model, discrete, target_update_interval, tau: float = 1):
        self.discrimination_model = discrimination_model
        self.discrete = discrete
        self.entropy_queue = deque(maxlen=10)
        self.entropy = None
        self.ensemble_size = self.discrimination_model.q_net.action_space.n
        self.num_timesteps = 0
        self.target_update_interval = target_update_interval
        self.tau = tau

    def train_step(self, batch, **kwargs):
        # we use the "q_net" output to get action probabilities
        logits = self.discrimination_model.q_net(batch.observations)

        # Calculate prediction loss and equalize across classes
        loss_vec = torch.nn.CrossEntropyLoss(reduction='none')(logits, batch.members.view(-1))
        one_hot = torch.nn.functional.one_hot(batch.members.view(-1), self.ensemble_size)
        loss_mat = loss_vec.unsqueeze(1) * one_hot
        total_per_class = one_hot.sum(0) + 1e-8
        total_loss_per_class = loss_mat.mean(0)
        class_mean_loss = total_loss_per_class / total_per_class
        loss = class_mean_loss.mean()

        # Measure class accuracy
        hits_vec = (logits.argmax(dim=1) == batch.members.view(-1)).float()
        total_class_hits = hits_vec.unsqueeze(1) * one_hot
        acc_vec = total_class_hits.sum(0) / total_per_class

        # Optimize the discrimination model
        self.discrimination_model.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.discrimination_model.parameters(), kwargs.get('max_grad_norm', np.inf))
        self.discrimination_model.optimizer.step()

        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.discrimination_model.q_net.parameters(), self.discrimination_model.q_net_target.parameters(), self.tau)

        logger.record("discrimination model/loss", loss.item())
        logger.record("discrimination model/accuracy", format_string(acc_vec.cpu().numpy().tolist()))
