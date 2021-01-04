from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.dqn import DQN
import types
from ensemble.common.buffers import EnsembleReplayBuffer
from ensemble.common.collect_rollout import collect_rollouts
from ensemble.common.sample_action import sample_action
from ensemble.common.entropy import HLoss
from ensemble.common.format_string import format_string
from torch.distributions.categorical import Categorical


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
            target_entropy: Union[str, float] = "auto",
            ent_coef: float = 0.1,
            max_ent_frac: float = 1.,
            max_ent_coef: float = 1.,
            min_ent_coef: float = 0.001,
            method: str = 'none',
            temperature: float = 1,
            soft: bool = True,
            ensemble_size: int = 1,
    ):
        policy_kwargs.update({"ensemble_size": ensemble_size,
                              "temperature": temperature,
                              "soft": soft})

        self.discrimination_trainer = discrimination_trainer
        self.ent_coef = ent_coef
        self.ent_coef_optimizer = None
        self.max_ent_frac = max_ent_frac
        self.max_ent_coef = max_ent_coef
        self.min_ent_coef = min_ent_coef
        self.target_entropy = target_entropy
        self.method = method
        self.temperature = temperature
        self.soft = soft
        self.ensemble_size = ensemble_size

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

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            b, c, h, w = replay_data.observations.shape

            ent_coef = self.ent_coef

            # train discrimination model
            if self.method == 'state':
                self.discrimination_trainer.train_step(replay_data, max_grad_norm=self.max_grad_norm)

                next_member_logits = self.discrimination_trainer.discrimination_model.q_net_target(
                    replay_data.next_observations)

                m = Categorical(logits=next_member_logits)
                disc_ent = m.entropy().mean()
                logger.record("discrimination model/ent", disc_ent.item())

                # Auto adjust ent coefficient by measuring state model entropy
                ent_coef_loss = None
                if self.ent_coef_optimizer is not None:
                    # Important: detach the variable from the graph
                    ent_coef = th.exp(self.log_ent_coef.detach())
                    # clipping entropy coefficient from above
                    ent_coef = th.clamp(ent_coef, self.min_ent_coef, self.max_ent_coef)
                    ent_coef_loss = - self.log_ent_coef * (disc_ent - self.target_entropy)

                # Optimize entropy coefficient, also called
                # entropy temperature or alpha in the paper
                if ent_coef_loss is not None:
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

            # Get current Q estimates
            current_q = self.q_net(replay_data.observations).view(b, self.ensemble_size, -1)

            with th.no_grad():
                # Compute the target Q values
                target_q = self.q_net_target(replay_data.next_observations).view(b, self.ensemble_size, -1)

                next_pi_logits = target_q / self.temperature
                if self.soft:
                    # Follow softmax policy
                    next_pi = th.nn.Softmax(dim=2)(next_pi_logits)
                    target_q = (next_pi * target_q).sum(-1)
                else:
                    target_q, _ = target_q.max(dim=2)
                # Ensemble Entropy Regularization
                ent = HLoss()(next_pi_logits, dim=2)
                if self.method == 'plain':
                    g = th.zeros_like(target_q)
                elif self.method == 'entropy':
                    g = ent
                elif self.method == 'action':
                    pi = th.nn.Softmax(dim=2)(current_q / self.temperature)
                    z = -th.log(pi + 1e-10)
                    a = z.cumsum(1) - z + 1e-8
                    idxs = replay_data.actions.long().repeat(1, self.ensemble_size).view(b, self.ensemble_size, 1)
                    g = th.gather(a, dim=2, index=idxs).squeeze(2)
                    w = (1. / th.arange(1, self.ensemble_size + 1, device=self.device)).unsqueeze(0)
                    g = g * w
                elif self.method == 'next_action':
                    next_pi_repeated = next_pi.repeat(1, self.ensemble_size, 1)
                    next_pi_interleaved = th.repeat_interleave(next_pi, repeats=self.ensemble_size, dim=1)
                    z = th.nn.KLDivLoss(reduction='none')(input=(next_pi_interleaved + 1e-8).log(), target=next_pi_repeated).sum(2)
                    z = z.view(b, self.ensemble_size, self.ensemble_size)
                    g = th.cat([th.tril(x, diagonal=-1).sum(1, keepdims=True).view(1, -1) for x in z], 0)
                    w = (1. / th.arange(1, self.ensemble_size + 1, device=self.device)).unsqueeze(0)
                    g = g * w
                elif self.method == 'state':
                    next_member_logprob = th.nn.LogSoftmax(dim=1)(next_member_logits)
                    cum_next_member_logprob = next_member_logprob.cumsum(1) - next_member_logprob
                    g = - cum_next_member_logprob
                    w = (1. / th.arange(1, self.ensemble_size + 1, device=self.device)).unsqueeze(0)
                    g = g * w
                else:
                    raise ValueError

                # Apply Entropy regularization
                target_q += ent_coef * g

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
            logger.record("train/soft", self.soft, exclude="tensorboard")
            logger.record("train/entropy", format_string(ent.mean(0).cpu().numpy().tolist()), exclude="tensorboard")
            logger.record("train/ent_coef", format_string(th.tensor(ent_coef).cpu().numpy().tolist()))
            logger.record("train/g", format_string(g.mean(0).cpu().numpy().tolist()), exclude="tensorboard")
            logger.record("train/Q", format_string(current_q.mean(0).cpu().detach().numpy().tolist()), exclude="tensorboard")
            logger.record("train/loss_vec", format_string(loss_vec.cpu().detach().numpy().tolist()), exclude="tensorboard")

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        if not deterministic and np.random.rand() < self.exploration_rate:
            action = [[self.action_space.sample() for _ in range(self.ensemble_size)]]
        else:
            action, state = self.policy.predict(observation, deterministic=deterministic)
            action = [action]
        return action, state

    def _setup_model(self) -> None:

        super(MaxEntDQN, self)._setup_model()

        # Target entropy (of the state model) is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = self.max_ent_frac * np.log(self.ensemble_size).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            assert self.method == 'state', f"Auto coefficient tuning is not supported in {self.method} mode!!!"
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)

            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)