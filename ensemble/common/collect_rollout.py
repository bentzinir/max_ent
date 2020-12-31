from typing import Optional

import numpy as np

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import logger
from ensemble.common.format_string import format_string


def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_episodes: int = 1,
        n_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        replay_buffer: Optional[ReplayBuffer] = None,
        log_interval: Optional[int] = None,
) -> RolloutReturn:
    episode_rewards, total_timesteps = [], []
    total_steps, total_episodes = 0, 0

    assert isinstance(env, VecEnv), "You must pass a VecEnv"
    assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

    if self.use_sde:
        self.actor.reset_noise()

    callback.on_rollout_start()
    continue_training = True

    while total_steps < n_steps or total_episodes < n_episodes:
        done = False
        episode_reward, episode_timesteps = 0.0, 0

        while not done:

            if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise()

            # Select action randomly or according to policy
            action, buffer_action = self._sample_action(learning_starts, action_noise)

            # Rescale and perform action
            new_obs, reward, done, infos = env.step(action)

            self.num_timesteps += 1
            episode_timesteps += 1
            total_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

            episode_reward += reward

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, done)

            # Store data in replay buffer
            if replay_buffer is not None:
                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs()
                    reward_ = self._vec_normalize_env.get_original_reward()
                else:
                    # Avoid changing the original ones
                    self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

                replay_buffer.add(self._last_original_obs, new_obs_, buffer_action, reward_, done, env.member)

            self._last_obs = new_obs
            # Save the unnormalized observation
            if self._vec_normalize_env is not None:
                self._last_original_obs = new_obs_

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is done as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            if 0 < n_steps <= total_steps:
                break

        if done:
            total_episodes += 1
            self._episode_num += 1
            episode_rewards.append(episode_reward)
            total_timesteps.append(episode_timesteps)

            if action_noise is not None:
                action_noise.reset()

            # Log training infos
            if log_interval is not None and self._episode_num % log_interval == 0:
                self._dump_logs()

            # calculate how much each member is played (relevant for prioritized mode)
            h = np.histogram(env.member_hist, bins=range(self.ensemble_size + 1))[0]
            h = h / h.sum()
            logger.record("train/rewards",
                          format_string([np.nanmean(env.reward_queues[idx]) for idx in range(self.ensemble_size)]),
                          exclude="tensorboard")
            logger.record("train/ep_len",
                          format_string([np.nanmean(env.eplen_queues[idx]) for idx in range(self.ensemble_size)]),
                          exclude="tensorboard")
            logger.record("train/member_hist", format_string(h.tolist()), exclude="tensorboard")
            logger.record("train/ID", self.env.unwrapped.envs[0].spec.id, exclude="tensorboard")
    mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

    callback.on_rollout_end()

    return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)
