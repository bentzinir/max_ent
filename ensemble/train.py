import gym
import envs
import ensemble.dqn.policies
import numpy as np
import time
from tqdm import tqdm
from ensemble.dummy_ensemble_vec_env import DummyEnsembleVecEnv
from ensemble.discriminator_trainer import DiscriminatorTrainer
from gym import spaces
import torch
import GPUtil


def eval_policy(env, model, steps=1000):
    obs = env.reset()
    traj_rewards = [0]
    while True:
        if np.random.rand() < model.exploration_rate:
            action = [model.action_space.sample() for _ in range(model.ensemble_size)]
        else:
            action, _state = model.policy.predict(obs, deterministic=False)
            action = action
        next_obs, reward, done, info = env.step([action])
        obs = next_obs
        env.render()
        time.sleep(0.03)
        traj_rewards[-1] += reward
        if done:
            obs = env.reset()
            m = input('Enter member idx: ')
            env.member = int(m)
            print(f"env member: {env.member}, R: {np.mean(traj_rewards)}")
            traj_rewards.append(0)


def train():
    if torch.cuda.is_available():
        deviceIds = GPUtil.getFirstAvailable(order='memory', maxLoad=0.8, maxMemory=0.8)
        device = torch.device(f'cuda:{deviceIds[0]}')
    else:
        device = torch.device('cpu')

    lr = 7e-4
    gamma = 0.9
    buffer_size = 50000
    batch_size = 128
    learning_starts = 10000
    total_timesteps = 150000
    temperature = 0.02
    exploration_final_rate = 0.05
    exploration_initial_rate = 0.05
    target_update_interval = 100
    log_interval = 50
    # Regularization types:
    # 1. none: g = 0
    # 2. entropy: g = entropy
    # 3. mutual_info:
    # 4. state: g = - log discrimination
    # method = 'none'; ent_coef = 0
    # method = 'entropy'; ent_coef = 0.05
    # method = 'next_action'; ent_coef = 0.1
    # method = 'action'; ent_coef = 0.05
    # method = 'state'; ent_coef = 0.025
    ensemble_size = 4
    prioritized = True
    discrete = True
    empty = False
    n_redundancies = 1
    max_repeats = 1
    room_size = 10
    up_wind = 0.0
    down_wind = 0.0
    right_wind = 0.0
    left_wind = 0.0

    env = DummyEnsembleVecEnv([lambda: gym.make('rooms-v0', rows=room_size, cols=room_size, discrete=discrete,
                                                goal=[1, 1], state=[room_size - 2, room_size - 2],
                                                fixed_reset=True, n_redundancies=n_redundancies, max_repeats=max_repeats,
                                                horz_wind=(right_wind, left_wind), vert_wind=(up_wind, down_wind),
                                                empty=empty, seed=0)],
                              ensemble_size=ensemble_size, prioritized_ensemble=prioritized)

    obs_shape = list(env.observation_space.shape)
    if discrete:
        from ensemble.max_ent_dqn import MaxEntDQN as Algorithm
        from stable_baselines3.dqn.policies import CnnPolicy as DiscriminationModel
        disc_obs_shape = (obs_shape[2], *obs_shape[:2])
        policy = 'EnsembleCnnPolicy'

    else:
        from max_ent_sac import MaxEntSAC as Algorithm
        # from stable_baselines3.sac import MlpPolicy as Model
        from continuous_action_model import DiagGaussianPolicy as Model
        policy = 'MlpPolicy'
        cat_dim = 1

    # create action model obs space by extending env's obs space
    disc_obs_space = gym.spaces.Box(low=env.observation_space.low.min(),
                                       high=env.observation_space.high.max(),
                                       shape=disc_obs_shape,
                                       dtype=env.observation_space.dtype)
    if method == 'state':
        discrimination_model = DiscriminationModel(observation_space=disc_obs_space,
                                                   action_space=spaces.Discrete(ensemble_size),
                                                   lr_schedule=lambda x: lr).to(device)
        discrimination_trainer = DiscriminatorTrainer(discrimination_model=discrimination_model, discrete=discrete,
                                                      lr=lr)
    else:
        discrimination_trainer = None

    model = Algorithm(policy, env, verbose=1, gamma=gamma, buffer_size=buffer_size, learning_starts=learning_starts,
                      discrimination_trainer=discrimination_trainer, device=device,
                      ent_coef=ent_coef, method=method, temperature=temperature,
                      batch_size=batch_size, exploration_final_eps=exploration_final_rate,
                      exploration_initial_eps=exploration_initial_rate, learning_rate=lr,
                      policy_kwargs={}, ensemble_size=ensemble_size, target_update_interval=target_update_interval)

    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    model.save("rooms")

    eval_res = eval_policy(env, model)
    print(f'Eval Result = {eval_res}')


if __name__ == '__main__':
    train()
