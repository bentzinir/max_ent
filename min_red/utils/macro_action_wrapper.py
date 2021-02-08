import gym
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
import itertools
from collections import deque


class MacroActionRepeatEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int = 1, vis: bool = False):

        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._k = k

        self.macro_actions = list(itertools.product(range(env.action_space.n), repeat=k))
        self.action_space = gym.spaces.Discrete(len(self.macro_actions))
        print(f" !!!!!!!!! (Macro Action Wrapper). |A| :{env.action_space.n} Macro Length: {k}, Augmented |A|: {self.action_space.n} !!!!!!!!! ")
        assert self.action_space.n <= 64, "Augmented action space is too big"
        self.vis = vis
        self.num_timesteps = 0
        self.ep_info_buffer = deque(maxlen=100)

    def step(self, action: int) -> GymStepReturn:
        total_reward = 0.0
        done = None
        for a in self.macro_actions[action]:
            obs, reward, done, info = self.env.step(a)
            total_reward += reward
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
    ):
        if macro_length > 1:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            env = MacroActionRepeatEnv(env, k=macro_length, vis=vis)
        super(MacroActionWrapper, self).__init__(env)
