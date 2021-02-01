import os
import time
import argparse


def run(envs, ensemble_sizes, methods, n_repeats, wandb_log_interval, total_timesteps, pause):

    step_mixture = 'true'
    buffer_size = 500000
    log_interval = 1  # (episodes)

    for trials in range(n_repeats):
        for env in envs:
            for method in methods:
                for ens_size in ensemble_sizes:
                    cmd_line = f"python -m mixture.train --f=mixture/config/mujoco " \
                               f" --wrapper_kwargs.wandb_log_interval {wandb_log_interval} " \
                               f" --total_timesteps {total_timesteps} " \
                               f" --algorithm.learn.log_interval {log_interval} " \
                               f" --step_mixture {step_mixture}" \
                               f" --ensemble_size {ens_size} " \
                               f" --algorithm.policy.buffer_size {buffer_size} " \
                               f" --method {method} " \
                               f" --env_id {env} & "
                    print(cmd_line)
                    os.system(cmd_line)
                    time.sleep(pause)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+",
                        default=['Hopper-v2', 'HalfCheetah-v2', 'Ant-v2', 'Swimmer-v2', 'Walker2d-v2', 'Humanoid-v2'])
    parser.add_argument("--ensemble_sizes", nargs="+", default=[3])
    parser.add_argument("--methods", nargs="+", default=['mixture', 'semi-mixture', 'entropy'])
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--wandb_log_interval", type=int, default=10000)
    parser.add_argument("--total_timesteps", type=int, default=2000000)
    parser.add_argument("--pause", type=float, default=0.1)

    args, extra_args = parser.parse_known_args()
    run(**args.__dict__)
