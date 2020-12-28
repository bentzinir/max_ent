from stable_baselines3.common import logger
import torch
from torch.distributions.categorical import Categorical


class DiscriminatorTrainer:
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, discrimination_model, lr, discrete):
        self.discrimination_model = discrimination_model
        self.optimizer = torch.optim.Adam(self.discrimination_model.parameters(), lr=lr)
        self.discrete = discrete

    def train_step(self, batch, **kwargs):
        if self.discrete:
            self.train_step_discrete(batch, **kwargs)
        else:
            self.train_step_continuous(batch)

    def train_step_discrete(self, batch, max_grad_norm):
        # we use the "q_net" output to get action probabilities
        logits = self.discrimination_model.q_net(batch.observations)
        loss = torch.nn.CrossEntropyLoss()(logits, batch.members.view(-1))
        m = Categorical(logits=logits)

        # Optimize the discrimination model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.discrimination_model.parameters(), max_grad_norm)
        self.optimizer.step()
        acc = (logits.argmax(dim=1) == batch.members.view(-1)).float().mean()
        logger.record("discrimination model/loss", loss.item())
        logger.record("discrimination model/accuracy", acc.item())
        logger.record("discrimination model/entropy", torch.mean(m.entropy()).item())

    def train_step_continuous(self, batch):
        # 1. build s,s'=f(s,a) distribution function
        x = torch.cat((batch.observations, batch.next_observations), dim=self.cat_dim).float()
        # 1.1 calculate mu, sigma of the Gaussian action model
        mu, log_std, _ = self.action_model.actor.get_action_dist_params(x)
        # 1.2 update probability distribution with calculated mu, log_std
        self.action_model.actor.action_dist.proba_distribution(mu, log_std)
        # 2 use N(ss') to calculate the probability of actually played action
        a_logp = self.action_model.actor.action_dist.log_prob(batch.actions)

        # Optimize the action model
        loss = -a_logp.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.record("action model/loss", loss.item())
