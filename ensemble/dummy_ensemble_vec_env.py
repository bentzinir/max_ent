from stable_baselines3.common.vec_env import DummyVecEnv
import scipy.special
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Union

import gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
import random
from collections import deque


class DummyEnsembleVecEnv(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]], ensemble_size: int = 1,
                 step_mixture: bool = False,
                 prioritized_ensemble: bool = False):
        super(DummyEnsembleVecEnv, self).__init__(env_fns)
        self.ensemble_size = ensemble_size
        self.member = random.choice(range(ensemble_size))
        self.step_mixture = step_mixture
        self.reward_queues = [deque(maxlen=50) for _ in range(self.ensemble_size)]
        self.eplen_queues = [deque(maxlen=50) for _ in range(self.ensemble_size)]
        self.cumulative_reward = 0
        self.episode_len = 0
        self.prioritized_ensemble = prioritized_ensemble
        self.member_hist = deque(maxlen=100)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            if self.step_mixture:
                self.member = self.draw_member()
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx][self.member]
            )
            self.cumulative_reward += self.buf_rews[env_idx]
            self.episode_len += 1
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
                self.reward_queues[self.member].append(self.cumulative_reward)
                self.eplen_queues[self.member].append(self.episode_len)
                self.cumulative_reward = 0
                self.episode_len = 0
                self.member = self.draw_member()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def reset(self) -> VecEnvObs:
        print(f"Inside reset !!!!")
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self.reward_queues[self.member].append(self.cumulative_reward)
            self.cumulative_reward = 0
            self._save_obs(env_idx, obs)
            self.member = random.choice(range(self.ensemble_size))
        return self._obs_from_buf()

    def draw_member(self):
        w = [np.nanmean(queue) + 1e-8 for queue in self.reward_queues]
        if np.any(np.isnan(w)) or not self.prioritized_ensemble:
            w = [1./self.ensemble_size] * self.ensemble_size
        else:
            r_norm = np.array(w) / np.array(w).max()
            w = scipy.special.softmax(r_norm)
        m = np.random.choice(range(self.ensemble_size), 1, p=w)[0]
        self.member_hist.append(m)
        return m
