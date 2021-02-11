import gym
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class DelayedRewardEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, dt: int = 0, vis: bool = False):
        gym.Wrapper.__init__(self, env)
        self.dt = dt
        self.vis = vis
        self.cum_r = 0
        self.nsteps = 0

    def step(self, action: int) -> GymStepReturn:
        obs, reward, done, info = self.env.step(action)
        self.nsteps += 1
        self.cum_r += reward
        r = 0
        if self.nsteps > self.dt:
            r = self.cum_r
            self.cum_r = 0
            self.nsteps = 0
        return obs, r, done, info

    def reset(self, **kwargs):
        self.nsteps = 0
        self.cum_r = 0
        return self.env.reset(**kwargs)


class DelayedRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        dt: int = 0,
        vis: bool = False,
        **kwargs
    ):
        env = DelayedRewardEnv(env, dt=dt, vis=vis)
        super(DelayedRewardWrapper, self).__init__(env)
