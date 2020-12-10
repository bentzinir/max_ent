from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import logger
import torch
from torch.distributions.categorical import Categorical


class ActionModelTrainer:
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, action_model, lr):
        self.action_model = action_model
        self.optimizer = torch.optim.Adam(self.action_model.parameters(), lr=lr)

    def train_step(self, batch, max_grad_norm):
        """
        This event is triggered before updating the policy.
        """

        x = torch.cat((batch.observations, batch.next_observations), dim=-1)
        predicted = self.action_model(x)
        action_probs = torch.softmax(predicted, dim=-1)
        loss = torch.nn.CrossEntropyLoss()(action_probs, batch.actions.view(-1))
        m = Categorical(action_probs)

        # Optimize the action model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), max_grad_norm)
        self.optimizer.step()
        logger.record("action model/loss", loss.item())
        logger.record("action model/entropy", torch.mean(m.entropy()).item())
