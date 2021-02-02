import gym
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from collections import deque
import wandb
from stable_baselines3.common.utils import safe_mean


class WandbWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, wandb_log_interval: int = -1):

        gym.Wrapper.__init__(self, env)
        self.num_timesteps = 0
        self.ep_info_buffer = deque(maxlen=100)
        self.wandb_log_interval = wandb_log_interval

    def wandb_logging(self):
        if self.wandb_log_interval > 0 and self.num_timesteps % self.wandb_log_interval == 0:
            mean_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            wandb.log({f"reward": mean_reward}, step=self.num_timesteps)

    def step(self, action: int) -> GymStepReturn:
        self.wandb_logging()
        self.num_timesteps += 1
        observation, reward, done, info = self.env.step(action)
        if done:
            self.ep_info_buffer.append(info['episode'])
        return observation, reward, done, info