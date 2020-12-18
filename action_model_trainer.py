from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import logger
import torch
from torch.distributions.categorical import Categorical


class ActionModelTrainer:
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, action_model, cat_dim, lr, discrete):
        self.action_model = action_model
        self.optimizer = torch.optim.Adam(self.action_model.parameters(), lr=lr)
        self.cat_dim = cat_dim
        self.discrete = discrete

    def train_step(self, batch, **kwargs):
        if self.discrete:
            self.train_step_discrete(batch, **kwargs)
        else:
            self.train_step_continuous(batch)

    def train_step_discrete(self, batch, max_grad_norm):
        x = torch.cat((batch.observations, batch.next_observations), dim=self.cat_dim).float()
        # we use the "q_net" output to get action probabilities
        predicted = self.action_model.q_net(x)
        action_probs = torch.softmax(predicted, dim=-1)
        loss = torch.nn.CrossEntropyLoss()(action_probs, batch.actions.view(-1))
        m = Categorical(action_probs)

        # Optimize the action model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), max_grad_norm)
        self.optimizer.step()
        acc = ((action_probs.argmax(dim=1) == batch.actions[:, 0])).float().mean().item()
        logger.record("action model/loss", loss.item())
        logger.record("action model/accuracy", acc)
        logger.record("action model/entropy", torch.mean(m.entropy()).item())
        logger.record("action model/hist", torch.histc(batch.actions.float(), bins=action_probs.shape[1]).tolist())

    def train_step_continuous(self, batch):
        # 1. build s,s'=f(s,a) distribution function
        x = torch.cat((batch.observations, batch.next_observations), dim=self.cat_dim).float()
        # 1.1 calculate mu, sigma of the Gaussian action model
        mu, log_std, _ = self.action_model.actor.get_action_dist_params(x)
        # 1.2 update probability distribution with calculated mu, log_std
        self.action_model.actor.action_dist.proba_distribution(mu, log_std)
        # 2 use N(ss') to calculate the probability of actually played action
        a_logp = self.action_model.actor.action_dist.log_prob(batch.actions)

        loss = -a_logp.mean()
        # Optimize the action model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.record("action model/loss", loss.item())
