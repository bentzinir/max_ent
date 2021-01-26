import gym
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame, ClipRewardEnv
try:
    import cv2  # pytype:disable=import-error

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None

from gym.wrappers.frame_stack import FrameStack
import numpy as np
from gym.spaces import Box


class LoopFireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset.

        For environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        done = True
        reset_count = 0
        while done:
            reset_count += 1
            self.env.reset(**kwargs)
            obs, _, done, _ = self.env.step(1)
            if reset_count == 10:
                raise AssertionError
        return obs

    def step(self, ac):
        return self.env.step(ac)


class AtariTransposeChannels(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """
        transpose stacked k last channels to last dimension
        :param env: the environment to wrap
        """
        gym.Wrapper.__init__(self, env)

        num_stack, h, w, c = self.observation_space.shape

        low = self.observation_space.low.swapaxes(0, 3).squeeze(axis=0)
        high = self.observation_space.high.swapaxes(0, 3).squeeze(axis=0)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    @staticmethod
    def transpose(x):
        return np.transpose(x, [1, 2, 0, 3]).squeeze(axis=3)

    def _get_observation(self):
        return self.transpose(self.frames)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.transpose(observation), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.transpose(observation)


class AtariStackWrapper(gym.Wrapper):
    """
    Atari 2600 preprocessings

    Specifically:

    * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}

    :param env: gym environment
    :param noop_max:: max number of no-ops
    :param frame_skip:: the frequency at which the agent experiences the game.
    :param screen_size:: resize Atari frame
    :param terminal_on_life_loss:: if True, then step() returns done=True whenever a
            life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        num_stack: int = 4,
    ):
        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            # replacing with custom fire reset env
            # env = FireResetEnv(env)
            env = LoopFireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)
        # num_stack = 2
        # assert False, "Please set the num_stack properly"
        env = FrameStack(env, num_stack=num_stack)
        env = AtariTransposeChannels(env)
        super(AtariStackWrapper, self).__init__(env)
