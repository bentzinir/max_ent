import gym
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
import itertools
from collections import deque
import wandb
from stable_baselines3.common.utils import safe_mean

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
        self.ep_info_buffer = deque(maxlen=100)
        self.wandb_log_interval = wandb_log_interval

    def wandb_logging(self):
        if self.wandb_log_interval > 0 and self.num_timesteps % self.wandb_log_interval == 0:
            mean_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            wandb.log({f"reward": mean_reward}, step=self.num_timesteps)

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
                maybe_ep_info = info.get("episode")
                if maybe_ep_info is not None:
                    self.ep_info_buffer.extend([maybe_ep_info])
                break
        return obs, total_reward, done, info


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
