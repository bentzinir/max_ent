import gym
import envs
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

env = gym.make('rooms-v0', rows=16, cols=16, horz_wind=(0, 0), vert_wind=(0, 0))

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("rooms")

del model  # remove to demonstrate saving and loading

model = DQN.load("rooms")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()