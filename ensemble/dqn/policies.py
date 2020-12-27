from typing import Any, Callable, Dict, List, Optional, Type

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp

from stable_baselines3.dqn.policies import QNetwork, DQNPolicy


class EnsembleQNetwork(QNetwork):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = [64, 64],
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        ensemble_size: int = 1,
    ):
        super(EnsembleQNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
            normalize_images,
        )

        self.ensemble_size = ensemble_size
        del self.q_net
        action_dim = self.action_space.n  # number of actions
        # q_net = create_mlp(self.features_dim, action_dim * ensemble_size, self.net_arch, self.activation_fn)
        # self.q_net = nn.Sequential(*q_net)
        self.q_net = []
        for e in range(ensemble_size):
            z = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
            self.q_net.append(nn.Sequential(*z))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        x = self.extract_features(obs)
        return th.cat([q(x).unsqueeze(1) for q in self.q_net], dim=1)


class EnsembleDQNPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        net_arch: Optional[List[int]] = [64, 64],
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        ensemble_size: int = 1,
    ):
        self.ensemble_size = ensemble_size
        super(EnsembleDQNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return EnsembleQNetwork(**net_args, ensemble_size=self.ensemble_size).to(self.device)


EnsembleMlpPolicy = EnsembleDQNPolicy


class EnsembleCnnPolicy(EnsembleDQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        net_arch: Optional[List[int]] = [64, 64],
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        ensemble_size: int = 1,
    ):

        super(EnsembleCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            ensemble_size,
        )


register_policy("EnsembleMlpPolicy", EnsembleMlpPolicy)
register_policy("EnsembleCnnPolicy", EnsembleCnnPolicy)
