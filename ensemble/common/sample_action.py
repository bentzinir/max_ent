from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
from stable_baselines3.common.noise import ActionNoise


def sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample an action according to the exploration policy.
    This is either done by sampling the probability distribution of the policy,
    or sampling a random action (from a uniform distribution over the action space)
    or by adding noise to the deterministic output.

    :param action_noise: Action noise that will be used for exploration
        Required for deterministic policy (e.g. TD3). This can also be used
        in addition to the stochastic policy for SAC.
    :param learning_starts: Number of steps before learning for the warm-up phase.
    :return: action to take in the environment
        and scaled action that will be stored in the replay buffer.
        The two differs when the action space is not normalized (bounds are not [-1, 1]).
    """
    # Select action randomly or according to policy
    if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
        # Warmup phase
        unscaled_action = [[self.action_space.sample() for k in range(self.ensemble_size)]]

    else:
        # Note: when using continuous actions,
        # we assume that the policy uses tanh to scale the action
        # We use non-deterministic action in the case of SAC, for TD3, it does not matter
        # unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        if np.random.rand() < self.exploration_rate:
            n_batch = self._last_obs.shape[0]
            unscaled_action = [[self.action_space.sample() for k in range(self.ensemble_size)]]
        else:
            unscaled_action, state = self.policy.predict(self._last_obs)
            unscaled_action = [unscaled_action]

    # Rescale the action from [low, high] to [-1, 1]
    if isinstance(self.action_space, gym.spaces.Box):
        scaled_action = self.policy.scale_action(unscaled_action)

        # Add noise to the action (improve exploration)
        if action_noise is not None:
            scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

        # We store the scaled action in the buffer
        buffer_action = scaled_action
        action = self.policy.unscale_action(scaled_action)
    else:
        # Discrete case, no need to normalize or clip
        buffer_action = unscaled_action
        action = buffer_action
    return action, buffer_action
