import gym
import envs
import ensemble.dqn.policies
import ensemble.sac.policies
import numpy as np
import time
from ensemble.dummy_ensemble_vec_env import DummyEnsembleVecEnv
from ensemble.discriminator_trainer import DiscriminatorTrainer
from gym import spaces
from config.parser_args import get_config
from config.config import Config
import argparse
from stable_baselines3.common.env_util import make_vec_env
from ensemble.make_atari_stack_env import make_atari_stack_env


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


def train(config):
    if config.is_atari:
        make_env = make_atari_stack_env
    else:
        make_env = make_vec_env
    env = make_env(config.env_id, n_envs=1, seed=0, vec_env_cls=DummyEnsembleVecEnv,
                   vec_env_kwargs=config.vec_env_kwargs, env_kwargs=config.env_kwargs)

    obs_shape = list(env.observation_space.shape)
    if config.algorithm.discrete:
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
    if config.algorithm.policy.method == 'state':
        d_model = DiscriminationModel(observation_space=disc_obs_space,
                                      action_space=spaces.Discrete(config.ensemble_size),
                                      lr_schedule=lambda x: config.algorithm.policy.learning_rate).to(config.device)
        discrimination_trainer = DiscriminatorTrainer(discrimination_model=d_model, discrete=config.algorithm.discrete)
    else:
        discrimination_trainer = None

    model = Algorithm(policy, env, discrimination_trainer=discrimination_trainer, **config.algorithm.policy)

    model.learn(**config.algorithm.learn)
    eval_policy(env, model)
    model.save(config.env_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    args, extra_args = parser.parse_known_args()
    config = get_config(args.f)
    algorithm_config = Config(config.algorithm_type)
    config.merge({"algorithm": algorithm_config}, override=False)
    config.algorithm.buffer["ensemble_size"] = config.ensemble_size
    config.algorithm.policy["ensemble_size"] = config.ensemble_size
    config.vec_env_kwargs["ensemble_size"] = config.ensemble_size
    config.algorithm.policy["device"] = config.device
    train(config)
