import gym
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class AccumulatedRewardEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, dr: float = -999, vis: bool = False):
        gym.Wrapper.__init__(self, env)
        self.dr = dr
        self.vis = vis
        self.cum_r = 0
        self.nsteps = 0

    def step(self, action: int) -> GymStepReturn:
        obs, reward, done, info = self.env.step(action)
        self.cum_r += reward
        r = 0
        if self.cum_r > self.dr:
            r = self.cum_r
            self.cum_r = 0
        return obs, r, done, info

    def reset(self, **kwargs):
        self.cum_r = 0
        return self.env.reset(**kwargs)


class AccumulatedRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        dr: float = -999,
        vis: bool = False,
        **kwargs
    ):
        env = AccumulatedRewardEnv(env, dr=dr, vis=vis)
        super(AccumulatedRewardWrapper, self).__init__(env)
