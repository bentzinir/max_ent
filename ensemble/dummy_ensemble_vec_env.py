from stable_baselines3.common.vec_env import DummyVecEnv
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Union

import gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
import random
from collections import deque


class DummyEnsembleVecEnv(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]], ensemble_size: int = 1):
        super(DummyEnsembleVecEnv, self).__init__(env_fns)
        self.ensemble_size = ensemble_size
        self.member = random.choice(range(ensemble_size))
        self.reward_queues = [deque(maxlen=10) for _ in range(self.ensemble_size)]
        self.cumulative_reward = 0

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx][self.member]
            )
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
                self.reward_queues[self.member].append(self.cumulative_reward)
                self.cumulative_reward = 0
                self.member = random.choice(range(self.ensemble_size))
            self._save_obs(env_idx, obs)
            self.cumulative_reward += self.buf_rews[env_idx]
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
