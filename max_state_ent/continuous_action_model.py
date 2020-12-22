from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.sac.policies import Actor
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import DiagGaussianDistribution
import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)


class DiagGaussianActor(Actor):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            normalize_images: bool = True,
    ):
        super(DiagGaussianActor, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            clip_mean,
            normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        self.action_dist = DiagGaussianDistribution(action_dim)


class DiagGaussianPolicy(SACPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
    ):
        super(DiagGaussianPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DiagGaussianActor(**actor_kwargs).to(self.device)