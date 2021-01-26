from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
import numpy as np
from stable_baselines3.common.distributions import StateDependentNoiseDistribution, DiagGaussianDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, create_sde_features_extractor, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
    create_mlp,
    get_actor_critic_arch,
)

from stable_baselines3.sac.policies import SACPolicy, Actor


# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class EnsembleActor(Actor):
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
        ensemble_size: int = 1,
        shared_body: bool = False,
    ):
        super(EnsembleActor, self).__init__(
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

        self.ensemble_size = ensemble_size
        action_dim = get_action_dim(self.action_space)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        # latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        # self.latent_pi = nn.Sequential(*latent_pi_net)

        self.latent_pis = []
        self.mus = []
        self.log_stds = []
        z = create_mlp(features_dim, -1, net_arch, activation_fn)
        for e in range(ensemble_size):
            if not shared_body:
                z = create_mlp(features_dim, -1, net_arch, activation_fn)
            self.latent_pis.append(nn.Sequential(*z))
            self.mus.append(nn.Linear(last_layer_dim, action_dim))
            self.log_stds.append(nn.Linear(last_layer_dim, action_dim))
        self.latent_pi = th.nn.Sequential(*self.latent_pis)
        self.mu = th.nn.Sequential(*self.mus)
        self.log_std = th.nn.Sequential(*self.log_stds)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = [latent_pi(features).unsqueeze(1) for latent_pi in self.latent_pis]
        mean_actions = [mu(latent) for mu, latent in zip(self.mus, latent_pi)]
        mean_actions = th.cat(mean_actions, 1)

        # Unstructured exploration (Original implementation)
        log_std = [logstd(latent) for logstd, latent in zip(self.log_stds, latent_pi)]
        log_std = th.cat(log_std, 1)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        a, s = [], []
        for e in range(self.ensemble_size):
            actions_n_std = self.action_dist.log_prob_from_params(mean_actions[:, e, :], log_std[:, e, :], **kwargs)
            a.append(actions_n_std[0])
            s.append(actions_n_std[1])
        return th.stack(a, dim=1), th.stack(s, dim=1), mean_actions, log_std

    def mixture_p(self, mu, log_std, actions, logp):
        mixture_p = th.zeros((mu.shape[0], self.ensemble_size), dtype=th.float).to(self.device)
        for e in range(self.ensemble_size):
            for j in range(self.ensemble_size):
                if j == e:
                    logp_e_j = logp[:, e]
                else:
                    self.action_dist.proba_distribution(mu[:, j, :], log_std[:, j, :])
                    logp_e_j = self.action_dist.log_prob(actions[:, e, :])
                    if th.isnan(logp_e_j).any():
                        print(f"Nan")
                logp_e_j = th.clamp(logp_e_j, np.log(1e-10), 40)
                mixture_p[:, e] += th.exp(logp_e_j)
        return mixture_p


class ContinuousEnsembleCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        ensemble_size: int = 1,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor,
        )

        action_dim = get_action_dim(self.action_space)
        self.ensemble_size = ensemble_size
        self.q_networks = []
        self.ensemble_dict = {}
        for e in range(self.ensemble_size):
            self.ensemble_dict[e] = {}
            for idx in range(n_critics):
                q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
                q_net = nn.Sequential(*q_net)
                self.add_module(f"qf{e}{idx}", q_net)
                self.q_networks.append(q_net)
                self.ensemble_dict[e][idx] = q_net

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        features = features.unsqueeze(1).repeat(1, self.ensemble_size, 1)
        qvalue_input = th.cat([features, actions], dim=2)
        outputs = []
        for e in range(self.ensemble_size):
            outputs_e = []
            qvalue_e = qvalue_input[:, e, :]
            for idx in range(self.n_critics):
                outputs_e.append(self.ensemble_dict[e][idx](qvalue_e))
            outputs.append(th.cat(outputs_e, 1).unsqueeze(1))
        return th.cat(outputs, 1)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        assert False, "Not implemented"
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features, actions], dim=1))


class EnsembleSACPolicy(SACPolicy):
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
        ensemble_size: int = 1,
        shared_critic: bool = False,
        shared_body: bool = False,
    ):
        self.ensemble_size = ensemble_size
        self.shared_critic = shared_critic
        self.shared_body = shared_body
        super(EnsembleSACPolicy, self).__init__(
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

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> EnsembleActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return EnsembleActor(**actor_kwargs, ensemble_size=self.ensemble_size, shared_body=self.shared_body).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) ->\
            Union[ContinuousCritic, ContinuousEnsembleCritic]:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        if self.shared_critic:
            return ContinuousCritic(**critic_kwargs).to(self.device)
        else:
            return ContinuousEnsembleCritic(**critic_kwargs, ensemble_size=self.ensemble_size).to(self.device)


EnsembleMlpPolicy = EnsembleSACPolicy

register_policy("EnsembleMlpPolicy", EnsembleMlpPolicy)
