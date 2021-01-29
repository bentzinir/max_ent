import gym
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
import itertools

try:
    import cv2  # pytype:disable=import-error

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None


class MacroActionRepeatEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int = 1, vis: bool = False):

        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._k = k
        self.macro_actions = list(itertools.permutations(range(env.action_space.n),k))
        self.action_space = gym.spaces.Discrete(len(self.macro_actions))
        self.vis = vis

    def step(self, action: int) -> GymStepReturn:
        total_reward = 0.0
        done = None
        for a in self.macro_actions[action]:
            obs, reward, done, info = self.env.step(a)
            total_reward += reward
            if self.vis:
                self.env.render()
            if done:
                break
        return obs, total_reward, done, info


class MacroActionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        macro_length: int = 1,
        vis: bool = False,
    ):
        env = MacroActionRepeatEnv(env, k=macro_length, vis=vis)
        super(MacroActionWrapper, self).__init__(env)
