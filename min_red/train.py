import gym
import envs
import numpy as np
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from min_red.action_model_trainer import ActionModelTrainer
from mixture.config.parser_args import get_config
from mixture.config.config import Config
import argparse
from stable_baselines3.common.env_util import make_vec_env
from mixture.make_atari_stack_env import make_atari_stack_env
from common.format_string import pretty
import wandb
import os


def eval_policy(env, model):
    obs = env.reset()
    traj_rewards = [0]
    while True:
        action, _state = model.predict(obs, deterministic=False)
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
    env = make_env(config.env_id, n_envs=1, seed=0, vec_env_cls=DummyVecEnv,
                   vec_env_kwargs=config.vec_env_kwargs, env_kwargs=config.env_kwargs)

    obs_shape = list(env.observation_space.shape)

    if config.algorithm.discrete:
        if config.algorithm.off_policy:
            from min_red.min_red_dqn import MinRedDQN as Algorithm
        else:
            from min_red.min_red_ppo import MinRedPPO as Algorithm
        from stable_baselines3.dqn.policies import CnnPolicy as ActionModel
        ssprime_shape = (2 * obs_shape[2], *obs_shape[:2])
        policy = 'CnnPolicy'
    else:
        from min_red.min_red_sac import MinRedSAC as Algorithm
        # from stable_baselines3.sac import MlpPolicy as Model
        from continuous_action_model import DiagGaussianPolicy as ActionModel
        ssprime_shape = (2*obs_shape[0],)
        policy = 'MlpPolicy'

    # create action model obs space by extending env's obs space
    ssprime_obs_space = gym.spaces.Box(low=env.observation_space.low.min(),
                                       high=env.observation_space.high.max(),
                                       shape=ssprime_shape,
                                       dtype=env.observation_space.dtype)

    action_model = ActionModel(observation_space=ssprime_obs_space,
                               action_space=env.action_space,
                               lr_schedule=lambda x: config.algorithm.policy.learning_rate).to(config.device)

    action_trainer = ActionModelTrainer(action_model=action_model,
                                        discrete=config.algorithm.discrete)

    model = Algorithm(policy, env, action_trainer=action_trainer, **config.algorithm.policy)

    model.learn(**config.algorithm.learn)
    print("Finished training...")
    if config.save_model:
        print("Saving model...")
        model.save(config.env_id)
    if config.play_model:
        eval_policy(env, model)


def bcast_config_vals(config):
    algorithm_config = Config(os.path.join(config.config_path, config.algorithm_type))
    config.merge({"algorithm": algorithm_config}, override=False)
    config.algorithm.learn.total_timesteps = config.total_timesteps
    config.algorithm.policy["device"] = config.device
    config.algorithm.policy.method = config.method
    config.algorithm.policy.wandb = config.wandb
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    args, extra_args = parser.parse_known_args()
    config = get_config(args.f)
    config = bcast_config_vals(config)
    if config.wandb:
        run = wandb.init(config=config)
    else:
        pretty(config)
    train(config)
