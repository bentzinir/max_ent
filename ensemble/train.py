import gym
import envs
import ensemble.dqn.policies
import ensemble.sac.policies
import numpy as np
import time
from ensemble.dummy_ensemble_vec_env import DummyEnsembleVecEnv
from ensemble.discriminator_trainer import DiscriminatorTrainer
from gym import spaces
import torch
import GPUtil
from config.parser_args import get_config
import argparse


def eval_policy(env, model):
    obs = env.reset()
    traj_rewards = [0]
    while True:
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, done, info = env.step(action)
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


def train(env_id, env_kwargs):
    if torch.cuda.is_available():
        deviceIds = GPUtil.getFirstAvailable(order='memory', maxLoad=0.8, maxMemory=0.8)
        config.device = torch.device(f'cuda:{deviceIds[0]}')
    else:
        config.device = torch.device('cpu')

    env = DummyEnsembleVecEnv([lambda: gym.make(env_id, **env_kwargs)], **config.buffer)
    obs_shape = list(env.observation_space.shape)
    if config.discrete:
        from ensemble.max_ent_dqn import MaxEntDQN as Algorithm
        from stable_baselines3.dqn.policies import CnnPolicy as DiscriminationModel
        disc_obs_shape = (obs_shape[2], *obs_shape[:2])
        policy = 'EnsembleCnnPolicy'
    else:
        from ensemble.max_ent_sac import MaxEntSAC as Algorithm
        from stable_baselines3.dqn.policies import MlpPolicy as DiscriminationModel
        disc_obs_shape = obs_shape
        policy = 'EnsembleMlpPolicy'

    # create action model obs space by extending env's obs space
    disc_obs_space = gym.spaces.Box(low=env.observation_space.low.min(),
                                    high=env.observation_space.high.max(),
                                    shape=disc_obs_shape,
                                    dtype=env.observation_space.dtype)
    if config.alg.method == 'state':
        discrimination_model = DiscriminationModel(observation_space=disc_obs_space,
                                                   action_space=spaces.Discrete(config.alg.ensemble_size),
                                                   lr_schedule=lambda x: config.alg.learning_rate).to(config.device)
        discrimination_trainer = DiscriminatorTrainer(discrimination_model=discrimination_model, discrete=config.discrete)
    else:
        discrimination_trainer = None

    model = Algorithm(policy, env, discrimination_trainer=discrimination_trainer, **config.alg)

    model.learn(**config.learn)
    eval_policy(env, model)
    model.save(env_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    args, extra_args = parser.parse_known_args()
    config = get_config("DQN")
    # config = get_config("SAC")
    config.buffer.ensemble_size = config.alg.ensemble_size

    env_id = 'rooms-v0'
    room_size = 10
    rooms_kwargs = {
        'n_redundancies': 4,
        'empty': False,
        'fixed_reset': True,
        'max_repeats': 1,
        'rows': room_size,
        'cols': room_size,
        'discrete': config.discrete,
        'goal': [1, 1],
        'state': [room_size - 2, room_size - 2],
        'horz_wind': (0.0, 0.0),  # (right, left)
        'vert_wind': (0.0, 0.0),  # (up, down)
        'seed': 0
    }

    train(env_id=env_id, env_kwargs=rooms_kwargs)
