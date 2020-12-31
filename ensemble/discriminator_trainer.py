from stable_baselines3.common import logger
import torch
from torch.distributions.categorical import Categorical
import numpy as np
from collections import deque


class DiscriminatorTrainer:
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, discrimination_model, discrete):
        self.discrimination_model = discrimination_model
        self.discrete = discrete
        self.entropy_queue = deque(maxlen=10)
        self.entropy = None

    def train_step(self, batch, **kwargs):
        # we use the "q_net" output to get action probabilities
        logits = self.discrimination_model.q_net(batch.observations)
        loss = torch.nn.CrossEntropyLoss()(logits, batch.members.view(-1))
        m = Categorical(logits=logits)
        self.entropy_queue.append(m.entropy().mean().item())
        self.entropy = np.mean(self.entropy_queue)

        # Optimize the discrimination model
        self.discrimination_model.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.discrimination_model.parameters(), kwargs.get('max_grad_norm', np.inf))
        self.discrimination_model.optimizer.step()
        acc = (logits.argmax(dim=1) == batch.members.view(-1)).float().mean()
        logger.record("discrimination model/loss", loss.item())
        logger.record("discrimination model/accuracy", acc.item())
        logger.record("discrimination model/ent", self.entropy)
