import gym
import envs
import numpy as np
import time
from max_ent_dqn import MaxEntDQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.torch_layers import NatureCNN, create_mlp
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
        # x = np.concatenate([obs, next_obs], axis=3)
        # a_prob = model.action_trainer.action_model(torch.tensor(x, device=model.device))
        # a_predicted = a_prob.detach().cpu().numpy()[0].argmax()
        # acc = 0.99 * acc + 0.01 * (action[0] == a_predicted)
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

spatial = True
lr = 2e-4
gamma = 0.95
buffer_size = 50000
batch_size = 64
learning_starts = 50000
total_timesteps = 250000
alpha = 0.01
exploration_final_rate = .05
active = True
n_repeats = 5

env = DummyVecEnv([lambda: gym.make('rooms-v0', rows=10, spatial=spatial, goal=[1, 1], state=[8, 8], fixed_reset=True,
                                    n_repeats=n_repeats, cols=10, empty=False, horz_wind=(0, 0), vert_wind=(0, 0), seed=0)])

# 1. Custom Cnn
if spatial:
    # 1.1 policy model
    policy = 'CnnPolicy'
    policy_kwargs = dict()
    # 1.2 action model
    nfeatures = 512
    h, w, c = env.observation_space.shape
    action_model_in_dim = (2 * c, h, w)
    cat_dim = 1
    action_model_obs_space = gym.spaces.Box(low=env.observation_space.low.min(),
                                            high=env.observation_space.high.max(),
                                            shape=action_model_in_dim,
                                            dtype=env.observation_space.dtype)
    action_model_features = NatureCNN(observation_space=action_model_obs_space, features_dim=nfeatures).to(device)
    action_model_mlp = create_mlp(nfeatures, env.action_space.n, net_arch=[])

    list_of_layers = list(action_model_features.children())
    list_of_layers.extend(action_model_mlp)
    action_model = nn.Sequential (*list_of_layers).to(device)

# 2.  MLP
else:
    # 2.1 policy model
    policy = MlpPolicy
    policy_kwargs = dict()
    # 2.2 action model
    action_model_obs_space = 2 * env.observation_space.shape[0]
    layer_dims = (action_model_obs_space, 64, env.action_space.n)
    layers = (nn.Linear,) * (len(layer_dims) - 1)
    cat_dim = 1
    action_model = MLP(layers=layers, layer_dims=layer_dims).to(device)

action_trainer = ActionModelTrainer(action_model=action_model, cat_dim=cat_dim, lr=lr)
model = MaxEntDQN(policy, env, verbose=1, gamma=gamma, buffer_size=buffer_size, learning_starts=learning_starts,
                  action_trainer=action_trainer, device=device, alpha=alpha, active=active, batch_size=batch_size,
                  exploration_final_eps=exploration_final_rate,
                  policy_kwargs=policy_kwargs)
model.learn(total_timesteps=total_timesteps, log_interval=100)
model.save("rooms")

eval_res = eval_policy(env, model, desc='Evaluating model')
print(f'Eval Result = {eval_res}')