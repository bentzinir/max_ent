import gym
import envs
import numpy as np
import time
from stable_baselines3 import DQN, PPO
from max_ent_dqn import MaxEntDQN
from stable_baselines3.dqn import MlpPolicy, CnnPolicy
from custom_cnn import CustomCnn
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv
from mlp import MLP
from action_model_trainer import ActionModelTrainer
import torch.nn as nn
import torch
import GPUtil


def eval_policy(env, model, steps=1000, desc=''):
    obs = env.reset()
    traj_rewards = [0]
    acc = 0
    for _ in tqdm(range(steps), desc=desc, leave=True):
        action, _state = model.predict(obs, deterministic=False)
        # print(f" time: {_}, action: {action}, obs.shape: {obs.shape}")
        next_obs, reward, done, info = env.step(action)
        x = np.concatenate([obs, next_obs], axis=-1)
        a_prob = model.action_trainer.action_model(torch.tensor(x, device=model.device))
        a_predicted = a_prob.detach().cpu().numpy()[0].argmax()
        acc = 0.99 * acc + 0.01 * (action[0] == a_predicted)
        obs = next_obs
        env.render()
        time.sleep(0.03)
        traj_rewards[-1] += reward
        # print(f"accuracy: {acc}")
        if done:
            obs = env.reset()
            traj_rewards.append(0)
    return np.mean(traj_rewards)


if torch.cuda.is_available():
    deviceIds = GPUtil.getFirstAvailable(order='memory', maxLoad=0.8, maxMemory=0.8)
    device = torch.device(f'cuda:{deviceIds[0]}')
else:
    device = torch.device('cpu')

env = DummyVecEnv([lambda: gym.make('rooms-v0', rows=10, spatial=False, goal=[1, 1],
                                    n_repeats=100, cols=10, empty=True, horz_wind=(0, 0), vert_wind=(0, 0), seed=0)])

# 1.  MLP

input_dim = 2 * env.observation_space.shape[0]
layer_dims = (input_dim, 64, env.action_space.n)
layers = (nn.Linear,) * (len(layer_dims) - 1)
action_model = MLP(layers=layers, layer_dims=layer_dims).to(device)
lr = 1e-3
action_trainer = ActionModelTrainer(action_model=action_model, lr=lr)
alpha = 0.01
beta = 0.0
model = MaxEntDQN(MlpPolicy, env, verbose=1, gamma=0.8, buffer_size=50000, learning_starts=50000,
                  action_trainer=action_trainer, device=device, alpha=alpha, beta=beta, batch_size=128)

# 2. Custom Cnn
# policy_kwargs = dict(features_extractor_class=CustomCnn)
# model = DQN('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1, gamma=0.8, buffer_size=10000)


# callback=None
model.learn(total_timesteps=150000, log_interval=100)
model.save("rooms")

eval_res = eval_policy(env, model, desc='Evaluating model')
print(f'Eval Result = {eval_res}')