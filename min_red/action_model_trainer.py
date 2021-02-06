from stable_baselines3.common import logger
import torch
from torch.distributions.categorical import Categorical
import gym
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
import torch as th
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.sac.policies import Actor


class ActionModelTrainer:
    def __init__(self, obs_space, act_space, lr, device, cat_dim=1):
        self.cat_dim = cat_dim
        self.discrete = isinstance(act_space, gym.spaces.Discrete)
        self.nupdates = 0

        obs_shape = list(obs_space.shape)
        ssprime_obs_space = gym.spaces.Box(
            low=obs_space.low.min(),
            high=obs_space.high.max(),
            shape=(2 * obs_shape[2], *obs_shape[:2]) if self.discrete else (2 * obs_shape[0],),
            dtype=obs_space.dtype)

        if self.discrete:
            self.action_model = CnnPolicy(
                observation_space=ssprime_obs_space,
                action_space=act_space,
                lr_schedule=lambda x: lr).to(device)
        else:
            self.action_model = Actor(
                observation_space=ssprime_obs_space,
                action_space=act_space,
                features_extractor=FlattenExtractor(ssprime_obs_space),
                net_arch=[256, 256],
                features_dim=ssprime_obs_space.shape[0]).to(device)

            self.action_dist = SquashedDiagGaussianDistribution(get_action_dim(self.action_model.action_space))
            self.optimizer = th.optim.Adam(self.action_model.parameters(), lr=lr)

    def train_step(self, batch, **kwargs):
        self.nupdates += 1
        if self.discrete:
            self.train_step_discrete(batch, **kwargs)
        else:
            self.train_step_continuous(batch)

    def train_step_discrete(self, batch, max_grad_norm):
        x = torch.cat((batch.observations, batch.next_observations), dim=self.cat_dim).float()
        # we use the "q_net" output to get action probabilities
        predicted = self.action_model.q_net(x)
        loss = torch.nn.CrossEntropyLoss()(predicted, batch.actions.view(-1))
        m = Categorical(logits=predicted)
        # Optimize the action model
        self.action_model.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), max_grad_norm)
        self.action_model.optimizer.step()
        acc = (predicted.argmax(dim=1) == batch.actions[:, 0]).float().mean().item()
        logger.record("action model/a_loss", loss.item())
        logger.record("action model/a_accuracy", acc)
        logger.record("action model/a_entropy", torch.mean(m.entropy()).item())
        logger.record("action model/a_hist", torch.histc(batch.actions.float(), bins=predicted.shape[1]).tolist())
        logger.record("action model/a_n_updates", self.nupdates)

    def train_step_continuous(self, batch):
        # 1. build s,s'=f(s,a) distribution function
        x = torch.cat((batch.observations, batch.next_observations), dim=self.cat_dim).float()
        # 1.1 calculate mu, sigma of the Gaussian action model
        mu, log_std, _ = self.action_model.get_action_dist_params(x)
        # 1.2 update probability distribution with calculated mu, log_std
        self.action_model.action_dist.proba_distribution(mu, log_std)
        # 2 use N(ss') to calculate the probability of actually played action
        a_logp = self.action_model.action_dist.log_prob(batch.actions)

        # Optimize the action model
        loss = -a_logp.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.record("action model/loss", loss.item())
