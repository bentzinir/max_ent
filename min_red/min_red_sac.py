from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac import SAC
import types
from min_red.utils.collect_rollouts import collect_rollouts
from min_red.utils.buffers import ISReplayBuffer
from min_red.min_red_regularization_stochastic import stochastic_min_red


class MinRedSAC(SAC):
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 3e-4,
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        n_episodes_rollout: int = -1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        action_trainer=None,
        alpha=0.1,
        method='none',
        regularization_starts: int = 1,
        importance_sampling: bool = False,
    ):

        self.action_trainer = action_trainer
        self.alpha = alpha
        self.method = method
        self.regularization_starts = regularization_starts
        self.importance_sampling = importance_sampling

        super(MinRedSAC, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            n_episodes_rollout,
            action_noise,
            optimize_memory_usage,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

        def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
            mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
            # return action and store log prob
            actions, log_prob = self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
            self._last_logp = log_prob
            assert not deterministic, "We thought that always deterministic=False"
            return actions

        # override actor's forward method for this object only
        self.actor.forward = types.MethodType(forward, self.actor)

        def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
            # CAP the standard deviation of the actor
            LOG_STD_MAX = 2
            LOG_STD_MIN = -5
            features = self.extract_features(obs)
            latent_pi = self.latent_pi(features)
            mean_actions = self.mu(latent_pi)

            if self.use_sde:
                latent_sde = latent_pi
                if self.sde_features_extractor is not None:
                    latent_sde = self.sde_features_extractor(features)
                return mean_actions, self.log_std, dict(latent_sde=latent_sde)
            # Unstructured exploration (Original implementation)
            log_std = self.log_std(latent_pi)
            # Original Implementation to cap the standard deviation
            log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            return mean_actions, log_std, {}

        # override action model minimum log_std
        self.action_trainer.action_model.actor.get_action_dist_params = types.MethodType(get_action_dist_params,
                                                                                   self.action_trainer.action_model.actor)

        # override collect_rollout method for this object only
        self.collect_rollouts = types.MethodType(collect_rollouts, self)

        self.actor._last_logp = th.from_numpy(np.asarray([0]))

        # override replay buffer
        self.replay_buffer = ISReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # train action model
            if self.method == 'stochastic':
                self.action_trainer.train_step(replay_data)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the target Q value: min over all critics targets
                targets = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                target_q, _ = th.min(targets, dim=1, keepdim=True)
                # add entropy term
                # OLD
                # target_q = target_q - ent_coef * next_log_prob.reshape(-1, 1)

                ###############################
                # NEW
                REWARDS = replay_data.rewards
                with th.no_grad():
                    if self.method == 'action':
                        g = - log_prob
                    elif self.method == 'stochastic':
                        g = stochastic_min_red(obs=replay_data.observations,
                                               next_obs=replay_data.next_observations,
                                               action=replay_data.actions,
                                               actor=self.actor,
                                               action_module=self.action_trainer.action_model.actor,
                                               logp_fresh=log_prob,
                                               logp0_replayed=replay_data.logp,
                                               cat_dim=self.action_trainer.cat_dim,
                                               importance_sampling=self.importance_sampling)
                    else:
                        raise ValueError

                    if th.any(th.isnan(g)).item():
                        print("Nan in g")

                    logger.record("action model/method", self.method, exclude="tensorboard")
                    logger.record("action model/g_mean", g.mean().item(), exclude="tensorboard")

                # td error + entropy term
                REWARDS = REWARDS + ent_coef * g.view(-1, 1)
                ###############################

                q_backup = REWARDS + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates for each critic network
            # using action from the replay buffer
            current_q_estimates = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, q_backup) for current_q in current_q_estimates])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        logger.record("action model/method", self.method, exclude="tensorboard")
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", np.mean(ent_coefs))
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

