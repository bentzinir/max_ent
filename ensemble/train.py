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


def eval_policy(env, model, steps=1000, desc=''):
    obs = env.reset()
    traj_rewards = [0]
    for _ in tqdm(range(steps), desc=desc, leave=True):
        action, _state = model.predict(obs, deterministic=False)
        next_obs, reward, done, info = env.step([action])
        obs = next_obs
        env.render()
        time.sleep(0.03)
        traj_rewards[-1] += reward
        if done:
            obs = env.reset()
            traj_rewards.append(0)
    return np.mean(traj_rewards)


def train():
    if torch.cuda.is_available():
        deviceIds = GPUtil.getFirstAvailable(order='memory', maxLoad=0.8, maxMemory=0.8)
        device = torch.device(f'cuda:{deviceIds[0]}')
    else:
        device = torch.device('cpu')

    lr = 7e-4
    gamma = 0.9
    buffer_size = 50000
    batch_size = 64
    learning_starts = 30000
    total_timesteps = 150000
    ent_coef = 0.01
    temperature = 0.01
    exploration_final_rate = 0
    exploration_initial_rate = 0
    # Regularization types:
    # 1. none: g = 0
    # 2. entropy: g = entropy
    # 3. ensemble_entropy: g = entropy + one sided kl
    # 4. state: g = - log discrimination
    method = 'ensemble_entropy'
    ensemble_size = 4
    discrete = True
    n_redundancies = 5
    max_repeats = 3
    room_size = 10
    up_wind = 0.0
    down_wind = 0.0
    right_wind = 0.0
    left_wind = 0.0

    env = DummyEnsembleVecEnv([lambda: gym.make('rooms-v0', rows=room_size, cols=room_size, discrete=discrete,
                                                goal=[1, 1], state=[room_size - 2, room_size - 2],
                                                fixed_reset=True, n_redundancies=n_redundancies, max_repeats=max_repeats,
                                                horz_wind=(right_wind, left_wind), vert_wind=(up_wind, down_wind),
                                                empty=False, seed=0, )], ensemble_size=ensemble_size)

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
                      exploration_initial_eps=exploration_initial_rate,
                      policy_kwargs={}, ensemble_size=ensemble_size)
    model.learn(total_timesteps=total_timesteps, log_interval=100)
    model.save("rooms")

    eval_res = eval_policy(env, model, desc='Evaluating model')
    print(f'Eval Result = {eval_res}')


if __name__ == '__main__':
    train()
