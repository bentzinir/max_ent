from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common import logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.dqn import DQN
import wandb
from min_red.min_red_regularization import action_probs, active_mask


class GroupedDQN(DQN):
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
            method: str = None,
            threshold: Union[None, float] = None,
            wandb: bool = True,
    ):

        super(GroupedDQN, self).__init__(
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

        self.wandb = wandb
        self.method = method
        self.action_trainer = action_trainer
        self.threshold = threshold

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # train action model
            if self.method == 'group':
                self.action_trainer.train_step(replay_data, max_grad_norm=None)

            n_actions = self.env.action_space.n

            # find equivalent actions
            if self.method == 'group':
                action_model_probs = action_probs(obs=replay_data.observations,
                                                  next_obs=replay_data.next_observations,
                                                  action_module=self.action_trainer.action_model.q_net,
                                                  cat_dim=self.action_trainer.cat_dim)

                active_action_mask = active_mask(actions=replay_data.actions,
                                                 action_model_probs=action_model_probs,
                                                 threshold=self.threshold)
            else:
                active_action_mask = F.one_hot(th.squeeze(replay_data.actions), n_actions).float()

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

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q, target_q.repeat(1, self.env.action_space.n), reduction='none')

            # Multiply loss matrix with active_action_mask
            loss = (loss * active_action_mask).sum(1)
            loss = loss.mean()
            losses.append(loss.item())
            mask_size = active_action_mask.sum(1).mean()

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
        logger.record("action model/mask_size", mask_size.item(), exclude="tensorboard")
        logger.record("action model/method", self.method, exclude="tensorboard")

        # wandb logging
        if self.wandb:
            rewards = [buf['r'] for buf in self.ep_info_buffer]
            wandb.log({f"reward": np.nanmean(rewards)}, step=self.num_timesteps)
