import gym
import envs
import numpy as np
import time
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from min_red.action_model_trainer import ActionModelTrainer
from mixture.config.parser_args import get_config
from mixture.config.config import Config
import argparse
from mixture.utils.make_atari_stack_env import make_atari_stack_env
from min_red.utils.make_wrapped_env import make_wrapped_env
from common.format_string import pretty
import wandb
import os
from datetime import date

try:
    import sparse_mujoco
except ImportError:
    print(f"sparse Mujoco not found")


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
        make_env = make_wrapped_env
    if config.algorithm.off_policy:
        vec_env = DummyVecEnv
        config.n_envs = 1
    else:
        vec_env = SubprocVecEnv

        # TODO: remove after running rooms-ppo experiment
        vec_env = DummyVecEnv
        config.n_envs = 1

    env = make_env(config.env_id, n_envs=config.n_envs, seed=0, vec_env_cls=vec_env,
                   wrapper_kwargs=config.wrapper_kwargs,
                   vec_env_kwargs=config.vec_env_kwargs,
                   env_kwargs=config.env_kwargs)

    if config.video_config.get("interval", 0) > 0:
        env = VecVideoRecorder(env, "videos",
                               record_video_trigger=lambda x: x % config.video_config.interval == 0,
                               video_length=config.video_config.length,
                               name_prefix=f"{config.env_id}_{date.today().strftime('%b-%d-%Y')}")

    # Algorithm
    if config.algorithm_type == 'GroupedQ':
        from min_red.grouped_dqn import GroupedDQN as Algorithm
    elif config.algorithm_type == 'DQN':
        from min_red.min_red_dqn import MinRedDQN as Algorithm
    elif config.algorithm_type == 'SAC':
        from min_red.min_red_sac import MinRedSAC as Algorithm
    elif config.algorithm_type == 'PPO':
        from min_red.min_red_ppo import MinRedPPO as Algorithm
    else:
        raise ValueError

    policy = 'CnnPolicy' if config.discrete else 'MlpPolicy'

    action_trainer = ActionModelTrainer(obs_space=env.observation_space,
                                        act_space=env.action_space,
                                        lr=config.algorithm.policy.learning_rate,
                                        device=config.device)

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
    config.macro_length = config.wrapper_kwargs.macro_length
    config.wandb_log_interval = config.wrapper_kwargs.wandb_log_interval
    config.algorithm.policy.wandb_log_interval = config.wrapper_kwargs.wandb_log_interval
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    args, extra_args = parser.parse_known_args()
    config = get_config(args.f)
    config = bcast_config_vals(config)
    if config.wandb_log_interval > 0:
        run = wandb.init(config=config)
    else:
        pretty(config)
    train(config)
