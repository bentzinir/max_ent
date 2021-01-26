from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import types
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common import logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.dqn import DQN
from stable_baselines3.common.distributions import Categorical
from min_red.utils.collect_rollouts import collect_rollouts
from min_red.utils.buffers import ISReplayBuffer


class MinRedDQN(DQN):
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
            soft: bool = False,
            ent_coef=0.1,
            method='none',
            importance_sampling: bool = False,
            absolute_threshold: bool = True,
            temperature=1,
            wandb: bool = True,
    ):

        super(MinRedDQN, self).__init__(
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

        def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
            q_values = self.forward(observation)
            # Greedy action
            if soft:
                pi = th.nn.Softmax(dim=1)(q_values / temperature)
                action = Categorical(probs=pi).sample()
            else:
                pi = q_values
                action = q_values.argmax(dim=1).reshape(-1)
            self._last_pi = pi[0][action[0]]
            return action

        self.wandb = wandb
        self.action_trainer = action_trainer
        self.ent_coef = ent_coef
        self.method = method
        self.importance_sampling = importance_sampling
        self.absolute_threshold = absolute_threshold
        self.temperature = temperature

        # override collect_rollout method for this object only
        self.collect_rollouts = types.MethodType(collect_rollouts, self)

        # override qnet_predict method for this object only
        self.q_net._predict = types.MethodType(_predict, self.q_net)
        self.q_net._last_pi = 1

        # override replay buffer
        self.replay_buffer = ISReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )

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
                # Follow softmax policy instead of eps-greedy
                next_pi = th.nn.Softmax(dim=1)(target_q / self.temperature)
                target_q = (next_pi * target_q).sum(-1)
                # Avoid potential broadcast issue
                target_q = target_q.reshape(-1, 1)
                # 1-step TD target
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates
            current_q = self.q_net(replay_data.observations)

            # Entropy & Action Uniqueness regularization
            x = th.cat((replay_data.observations, replay_data.next_observations), dim=self.action_trainer.cat_dim).float()
            action_model_logits = self.action_trainer.action_model.q_net(x)
            action_model_probs = th.nn.Softmax(dim=1)(action_model_logits)
            pi = th.nn.Softmax(dim=1)(current_q / self.temperature)
            a_mask = F.one_hot(th.squeeze(replay_data.actions), self.env.action_space.n).float()
            a_prime_mask = 1 - a_mask
            pi_a = th.sum(a_mask * pi, dim=1, keepdim=True)
            pa_a = th.sum(a_mask * action_model_probs, dim=1, keepdim=True)
            if self.absolute_threshold:
                thresh = 1e-2
            else:
                thresh = pa_a.repeat(1, self.env.action_space.n)
            active_actions = (action_model_probs >= thresh).float()
            pi_a_prime = th.sum(active_actions * a_prime_mask * pi, dim=1, keepdim=True)
            n_primes = th.mean(th.sum(active_actions * a_prime_mask, dim=1))
            logger.record("action model/n_primes", n_primes.item(), exclude="tensorboard")
            logger.record("action model/method", self.method, exclude="tensorboard")
            with th.no_grad():
                eps = 1e-4
                if self.method == 'none':
                    g = 0.0
                elif self.method == 'action':
                    g = - th.log(pi_a + eps)
                elif self.method == 'eta':
                    g = - th.log(pi_a + pi_a_prime + eps)
                    g = g
                elif self.method == 'stochastic':
                    g = th.log(pa_a + eps) - th.log(pi_a + eps)
                else:
                    raise ValueError
                if self.importance_sampling:
                    g = g * (pi_a / replay_data.pi)
            # Retrieve the q-values for the actions from the replay buffer
            current_q = th.gather(current_q, dim=1, index=replay_data.actions.long())

            # Apply Entropy regularization
            target_q += self.ent_coef * g

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
