from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.dqn import DQN
import types
from stable_baselines3.common.distributions import Categorical
from ensemble.common.buffers import EnsembleReplayBuffer
from ensemble.common.collect_rollout import collect_rollouts
from ensemble.common.sample_action import sample_action
from ensemble.common.entropy import HLoss
from ensemble.common.format_string import format_string


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
            discrimination_trainer=None,
            ent_coef: float = 0.1,
            method: str = 'none',
            temperature: float = 1,
            ensemble_size: int = 1,
    ):
        policy_kwargs.update({"ensemble_size": ensemble_size})
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

        def soft_predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
            q_values = self.forward(observation)
            # Greedy action
            if deterministic:
                return q_values.argmax(dim=1).reshape(-1)
            else:
                z = q_values / temperature
                return Categorical(logits=z).sample().squeeze(0)

        # replace self.q_net._predict with soft_predict for this object only
        self.q_net._predict = types.MethodType(soft_predict, self.q_net)

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

        self.discrimination_trainer = discrimination_trainer
        self.ent_coef = ent_coef
        self.method = method
        self.temperature = temperature
        self.ensemble_size = ensemble_size

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            b, c, h, w = replay_data.observations.shape

            # train discrimination model
            if self.method == 'state':
                self.discrimination_trainer.train_step(replay_data, max_grad_norm=self.max_grad_norm)

            # Get current Q estimates
            current_q = self.q_net(replay_data.observations).view(b, self.ensemble_size, -1)

            with th.no_grad():
                # Compute the target Q values
                target_q = self.q_net_target(replay_data.next_observations).view(b, self.ensemble_size, -1)

                # Follow softmax policy
                next_pi_logits = target_q / self.temperature
                next_pi = th.nn.Softmax(dim=2)(next_pi_logits)
                target_q = (next_pi * target_q).sum(-1)

                # Ensemble Entropy Regularization
                ent = HLoss()(next_pi_logits, dim=2)
                one_hot_active_idx = th.nn.functional.one_hot(replay_data.members.squeeze() + 1, self.ensemble_size + 1)
                child_mask = th.cumsum(one_hot_active_idx, dim=1)[:, :-1]
                if self.method == 'none':
                    g = th.tensor(0.0)
                elif self.method == 'entropy':
                    g = ent
                elif self.method == 'ensemble_entropy':
                    owner_one_hot = F.one_hot(replay_data.members.squeeze(), self.ensemble_size).unsqueeze(2)
                    owner_next_pi = (owner_one_hot * next_pi).sum(1).unsqueeze(1)
                    ce = - (next_pi * th.log(owner_next_pi)).sum(2)
                    descendants_ce = ce * child_mask
                    g = ent + descendants_ce
                    logger.record("train/cross_entropy", [descendants_ce.min().item(), descendants_ce.mean().item(),
                                                          descendants_ce.max().item()], exclude="tensorboard")
                elif self.method == 'mutual_info':
                    pi = th.nn.Softmax(dim=2)(current_q / self.temperature)
                    z = -th.log(pi + 1e-10)
                    a = z.cumsum(1) - z + 1e-8
                    idxs = replay_data.actions.long().repeat(1, self.ensemble_size).view(b, self.ensemble_size, 1)
                    g = th.gather(a, dim=2, index=idxs).squeeze(2)
                    w = (1. / th.arange(1, self.ensemble_size + 1, device=self.device)).unsqueeze(0)
                    g = g * w
                elif self.method == 'next_mutual_info':
                    next_pi_cumsum = next_pi.cumsum(1)
                    ens_next_pi = next_pi_cumsum / next_pi_cumsum.sum(2, keepdims=True)
                    # KLDivloss: input = logits. target = probs.
                    # g = th.nn.KLDivLoss(reduction='none')(input=(ens_next_pi+1e-2).log(), target=next_pi).sum(2)
                    g = th.nn.KLDivLoss(reduction='none')(input=(next_pi+1e-8).log(), target=ens_next_pi).sum(2)
                elif self.method == 'state':
                    next_member_logits = self.discrimination_trainer.discrimination_model.q_net(
                        replay_data.next_observations)
                    next_member_logprob = th.nn.LogSoftmax(dim=1)(next_member_logits)
                    # accumulate penalty from all masters
                    g = next_member_logprob - next_member_logprob.cumsum(1)
                    w = (1. / th.arange(1, self.ensemble_size + 1, device=self.device)).unsqueeze(0)
                    g = g * w
                else:
                    raise ValueError

                # Apply Entropy regularization
                target_q += self.ent_coef * g

                # 1-step TD target
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Retrieve the q-values for the actions from the replay buffer
            current_q = \
                th.gather(current_q, dim=2,
                          index=replay_data.actions.long().repeat(1, self.ensemble_size).view(b, self.ensemble_size, 1))

            # reshape to match target_q.shape
            current_q = current_q.reshape(b, self.ensemble_size)

            target_q = target_q.detach()
            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
            loss_vec = loss.mean(0)
            loss = loss_vec.mean()

            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Logging
            logger.record("train/method", self.method, exclude="tensorboard")
            logger.record("train/entropy", format_string(ent.mean(0).cpu().numpy().tolist()), exclude="tensorboard")
            logger.record("train/g", format_string(g.mean(0).cpu().numpy().tolist()), exclude="tensorboard")
            logger.record("train/Q", format_string(current_q.mean(0).cpu().detach().numpy().tolist()), exclude="tensorboard")
            logger.record("train/loss_vec", format_string(loss_vec.cpu().detach().numpy().tolist()), exclude="tensorboard")

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))
