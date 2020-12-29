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
from ensemble.common.collect_rollout import collect_rollouts
from ensemble.common.sample_action import sample_action
from ensemble.common.buffers import EnsembleReplayBuffer
from ensemble.common.format_string import format_string


class MaxEntSAC(SAC):
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
            discrimination_trainer=None,
            method: str = 'none',
            ensemble_size: int = 1,
    ):
        policy_kwargs.update({"ensemble_size": ensemble_size})
        super(MaxEntSAC, self).__init__(
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

        # override collect_rollout method for this object only
        self.collect_rollouts = types.MethodType(collect_rollouts, self)

        # override _sample_action method for this object only
        self._sample_action = types.MethodType(sample_action, self)

        # override replay buffer
        self.replay_buffer = EnsembleReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
            ensemble_size=ensemble_size,
        )

        self.method = method
        self.ensemble_size = ensemble_size
        self.discrimination_trainer = discrimination_trainer

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

            dones = replay_data.dones.repeat(1, self.ensemble_size).unsqueeze(2)
            rewards = replay_data.rewards.repeat(1, self.ensemble_size).unsqueeze(2)
            replayed_actions = replay_data.actions.unsqueeze(1).repeat(1, self.ensemble_size, 1)

            # train discrimination model
            if self.method == 'state':
                self.discrimination_trainer.train_step(replay_data)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, self.ensemble_size, 1)

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
                targets = self.critic_target(replay_data.next_observations, next_actions)
                target_q, _ = th.min(targets, dim=2, keepdim=True)
                # add entropy term
                if self.method == 'none':
                    g = th.zeros_like(target_q)
                elif self.method == 'entropy':
                    g = - next_log_prob.unsqueeze(2)
                elif self.method == 'action':
                    pass
                elif self.method == 'next_action':
                    ens_next_log_prob = next_log_prob.cumsum(1) - next_log_prob
                    w = (1. / th.arange(1, self.ensemble_size + 1, device=self.device)).unsqueeze(0)
                    g = - (ens_next_log_prob * w).unsqueeze(2)
                elif self.method == 'state':
                    next_member_logits = self.discrimination_trainer.discrimination_model.q_net(
                        replay_data.next_observations)
                    next_member_logprob = th.nn.LogSoftmax(dim=1)(next_member_logits)
                    # accumulate penalty from all masters
                    g = next_member_logprob - next_member_logprob.cumsum(1)
                    w = (1. / th.arange(1, self.ensemble_size + 1, device=self.device)).unsqueeze(0)
                    g = (g * w).unsqueeze(2)
                target_q = target_q + ent_coef * g
                # td error + entropy term
                q_backup = rewards + (1 - dones) * self.gamma * target_q

            # Get current Q estimates for each critic network
            # using action from the replay buffer
            current_q_estimates = self.critic(replay_data.observations, replayed_actions)

            # Compute critic loss
            # critic_loss = 0.5 * sum([F.mse_loss(current_q, q_backup) for current_q in current_q_estimates])
            critic_loss_vec = F.mse_loss(current_q_estimates, q_backup.repeat(1, 1, self.critic.n_critics),
                                         reduction='none').mean(0).sum(1)
            critic_loss = critic_loss_vec.mean()
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = self.critic(replay_data.observations, actions_pi)
            min_qf_pi, _ = th.min(q_values_pi, dim=2, keepdim=True)
            actor_loss_vec = (ent_coef * log_prob - min_qf_pi).mean(0).squeeze(1)
            actor_loss = actor_loss_vec.mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", np.mean(ent_coefs))
        logger.record("train/actor_loss", format_string(actor_loss_vec.cpu().detach().numpy().tolist()))
        logger.record("train/critic_loss", format_string(critic_loss_vec.cpu().detach().numpy().tolist()))
        logger.record("train/g", format_string(g.mean(0).squeeze(1).cpu().numpy().tolist()), exclude="tensorboard")
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

