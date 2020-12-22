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
            ent_coef=0.1,
            method='none',
            temperature=1,
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

        def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
            q_values = self.forward(observation)
            # Greedy action
            if deterministic:
                action = q_values.argmax(dim=1).reshape(-1)
            else:
                pi = th.nn.Softmax(dim=1)(q_values / temperature)
                action = Categorical(probs=pi).sample()
            return action

        # replace self.q_net._predict with _predict for this object only
        self.q_net._predict = types.MethodType(_predict, self.q_net)

        self.action_trainer = action_trainer
        self.ent_coef = ent_coef
        self.method = method
        self.temperature = temperature

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
                # target_q, _ = target_q.max(dim=1)
                # TODO: changing from Q-learning to SARSA-like training
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
            active_actions = (action_model_probs > 1e-2).float()
            pi_a_prime = th.sum(active_actions * a_prime_mask * pi, dim=1, keepdim=True)
            n_primes = th.mean(th.sum(active_actions * a_prime_mask, dim=1))
            logger.record("action model/n_primes", n_primes.item(), exclude="tensorboard")
            logger.record("action model/method", self.method, exclude="tensorboard")
            with th.no_grad():
                if self.method == 'none':
                    g = 0.0
                elif self.method == 'action':
                    g = - th.log(pi_a + 1e-2)
                elif self.method == 'next_det':
                    g = - th.log(pi_a + pi_a_prime + 1e-2)
                elif self.method == 'next_det_nir':
                    g = - pi_a * th.log(pi_a + pi_a_prime + 1e-2)
                elif self.method == 'next_abs':
                    g = th.abs(th.clamp(th.div(pa_a, pi_a + 1e-2), min=0.2, max=5) - 1)
                elif self.method == 'next_log':
                    g = th.log(th.clamp(th.div(pa_a, pi_a + 1e-2), min=0.2, max=5))
                else:
                    raise ValueError

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