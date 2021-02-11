import os
import time
import argparse


def run(delta, macro_length, methods, n_redundancies, n_repeats, wandb_log_interval, total_timesteps, pause, dry):

    log_interval = 10
    reg_starts = 10000
    buffer_size = 20000
    delta = None
    n_envs = 2

    for trials in range(n_repeats):
        for method in methods:
            for n_redun in n_redundancies:
                cmd_line = f"python -m min_red.train --f min_red/config/rooms --algorithm_type PPO " \
                           f" --wrapper_kwargs.wandb_log_interval {wandb_log_interval} " \
                           f" --method {method} " \
                           f" --env_kwargs.n_redundancies {n_redun} " \
                           f" --total_timesteps {total_timesteps} " \
                           f" --algorithm.learn.log_interval {log_interval} " \
                           f" --algorithm.policy.absolute_threshold True " \
                           f" --algorithm.policy.delta {delta} " \
                           f" --algorithm.policy.buffer_size {buffer_size}" \
                           f" --algorithm.policy.regularization_starts {reg_starts}" \
                           f" --wrapper_kwargs.macro_length {macro_length} " \
                           f" --n_envs {n_envs} " \
                           f" & "
                print(cmd_line)
                if not dry:
                    os.system(cmd_line)
                    time.sleep(pause)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--macro_length", type=int, default=1)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--methods", nargs="+", default=['eta', 'action'])
    parser.add_argument("--n_redundancies", nargs="+", default=[1])
    parser.add_argument("--wandb_log_interval", type=int, default=1000)
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--pause", type=float, default=0.1)
    parser.add_argument("--dry", action='store_true')

    args, extra_args = parser.parse_known_args()
    run(**args.__dict__)
