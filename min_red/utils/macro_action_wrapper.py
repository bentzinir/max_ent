import gym
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
import itertools
from collections import deque
import wandb
import numpy as np

try:
    import cv2  # pytype:disable=import-error

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None


class MacroActionRepeatEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int = 1, vis: bool = False, wandb_log_interval: int = -1):

        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._k = k
        self.macro_actions = list(itertools.permutations(range(env.action_space.n),k))
        self.action_space = gym.spaces.Discrete(len(self.macro_actions))
        self.vis = vis
        self.num_timesteps = 0
        self.reward_queue = deque(maxlen=50)
        self.cumulative_reward = 0
        self.wandb_log_interval = wandb_log_interval

    def wandb_logging(self):
        if self.wandb_log_interval > 0 and self.num_timesteps % self.wandb_log_interval == 0:
            ep_R = np.nanmean(self.reward_queue)
            wandb.log({f"reward": ep_R}, step=self.num_timesteps)

    def step(self, action: int) -> GymStepReturn:
        self.wandb_logging()

        total_reward = 0.0
        done = None
        for a in self.macro_actions[action]:
            obs, reward, done, info = self.env.step(a)
            total_reward += reward
            self.num_timesteps += 1
            if self.vis:
                self.env.render()
            if done:
                break
        self.cumulative_reward += total_reward
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        self.reward_queue.append(self.cumulative_reward)
        self.cumulative_reward = 0
        return self.env.reset(**kwargs)


class MacroActionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        macro_length: int = 1,
        vis: bool = False,
        wandb_log_interval: int = -1,
    ):
        env = MacroActionRepeatEnv(env, k=macro_length, vis=vis, wandb_log_interval=wandb_log_interval)
        super(MacroActionWrapper, self).__init__(env)
