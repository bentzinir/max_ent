import gym
import envs
import numpy as np
import time
from stable_baselines3 import DQN, PPO
from stable_baselines3.dqn import MlpPolicy, CnnPolicy
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv


def eval_policy(env, policy, steps=10000, desc=''):
    obs = env.reset()
    traj_rewards = [0]
    for _ in tqdm(range(steps), desc=desc, leave=True):
        action, _state = policy(obs, deterministic=False)
        # print(f" time: {_}, action: {action}, obs.shape: {obs.shape}")
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.03)
        traj_rewards[-1] += reward
        if done:
            obs = env.reset()
            traj_rewards.append(0)
    return np.mean(traj_rewards)


env = DummyVecEnv([lambda: gym.make('rooms-v0', rows=16, cols=16, horz_wind=(0, 0), vert_wind=(0, 0))])

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000000, log_interval=1000)
model.save("rooms")

eval_res = eval_policy(env, model.predict, desc='Evaluating model')
print(f'Eval Result = {eval_res}')