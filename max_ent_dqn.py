from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import get_linear_fn, polyak_update
from stable_baselines3.dqn.policies import DQNPolicy

from stable_baselines3.dqn import DQN

from types import SimpleNamespace


class MaxEntDQN(DQN):
    def __init__(
            self,
            policy: Union[str, Type[DQNPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Callable] = 1e-4,
            buffer_size: int = 1000000,
            learning_starts: int = 50000,
            batch_size: Optional[int] = 32,
            tau: float = 1.0,
            gamma: float = 0.99,
            train_freq: int = 4,
            gradient_steps: int = 1,
            n_episodes_rollout: int = -1,
            optimize_memory_usage: bool = False,
            target_update_interval: int = 10000,
            exploration_fraction: float = 0.1,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            max_grad_norm: float = 10,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            action_trainer=None,
            alpha=0.1,
            active=True,
            stochastic=False,
    ):

        super(MaxEntDQN, self).__init__(
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
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

        self.action_trainer = action_trainer
        self.alpha = alpha
        self.active = active
        self.stochastic = stochastic

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # train action model
            self.action_trainer.train_step(replay_data, max_grad_norm=self.max_grad_norm)

            with th.no_grad():
                # Compute the target Q values
                target_q = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                target_q, _ = target_q.max(dim=1)
                # Avoid potential broadcast issue
                target_q = target_q.reshape(-1, 1)
                # 1-step TD target
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates
            current_q = self.q_net(replay_data.observations)

            # Entropy & Action Uniqueness regularization
            x = th.cat((replay_data.observations, replay_data.next_observations), dim=self.action_trainer.cat_dim).float()
            action_model_logits = self.action_trainer.action_model(x)
            action_model_probs = th.nn.Softmax(dim=1)(action_model_logits)
            temperature = 0.1
            pi = th.nn.Softmax(dim=1)(temperature * current_q)
            a_mask = F.one_hot(th.squeeze(replay_data.actions), self.env.action_space.n).float()
            a_prime_mask = 1 - a_mask
            pi_a = th.sum(a_mask * pi, dim=1, keepdim=True)
            pa_a = th.sum(a_mask * action_model_probs, dim=1, keepdim=True)
            active_actions = (action_model_probs > 1e-2).float()
            pi_a_prime = th.sum(active_actions * a_prime_mask * pi, dim=1, keepdim=True)
            n_primes = th.mean(th.sum(active_actions * a_prime_mask, dim=1))
            logger.record("action model/n_primes", n_primes.item(), exclude="tensorboard")
            logger.record("action model/active", self.active, exclude="tensorboard")
            logger.record("action model/stochastic", self.stochastic, exclude="tensorboard")
            if self.active:
                if self.stochastic:
                    e = th.clamp(th.div(pi_a, pa_a), min=0.2, max=5)
                else:
                    e = pi_a + pi_a_prime
            else:
                e = pi_a
            # Retrieve the q-values for the actions from the replay buffer
            current_q = th.gather(current_q, dim=1, index=replay_data.actions.long())

            # Apply Entropy regularization
            target_q += -self.alpha * th.log(e).detach()

            target_q = target_q.detach()
            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q, target_q)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))
