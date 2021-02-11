import os
import time
import argparse


def run(envs, ent_coef, betas, dts, drs, learning_rate, learning_starts, methods, n_repeats, buffer_size, wandb_log_interval, total_timesteps, pause, dry):

    log_interval = 1

    for trials in range(n_repeats):
        for env in envs:
            for method in methods:
                for beta in betas:
                    for dt in dts:
                        for dr in drs:
                            cmd_line = f"python -m min_red.train --f min_red/config/mujoco --algorithm_type SAC " \
                                       f" --wrapper_kwargs.wandb_log_interval {wandb_log_interval} " \
                                       f" --method {method} " \
                                       f" --algorithm.policy.learning_rate {learning_rate} " \
                                       f" --algorithm.policy.learning_starts {learning_starts} " \
                                       f" --algorithm.policy.ent_coef {ent_coef} " \
                                       f" --algorithm.policy.beta {beta} " \
                                       f" --wrapper_kwargs.dt {dt} " \
                                       f" --wrapper_kwargs.dr {dr} " \
                                       f" --algorithm.policy.buffer_size {buffer_size} " \
                                       f" --total_timesteps {total_timesteps} " \
                                       f" --algorithm.learn.log_interval {log_interval} " \
                                       f" --env_id {env} & "
                            print(cmd_line)
                            print("\n")
                            if not dry:
                                os.system(cmd_line)
                                time.sleep(pause)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+",
                        default=['Hopper-v2', 'HalfCheetah-v2', 'Ant-v2', 'Swimmer-v2', 'Walker2d-v2', 'Humanoid-v2'])
    parser.add_argument("--methods", nargs="+", default=['action', 'stochastic'])
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--wandb_log_interval", type=int, default=10000)
    parser.add_argument("--ent_coef", type=float, default=0.1)
    parser.add_argument("--betas", nargs="+", default=[0.01])
    parser.add_argument("--dts", nargs="+", default=[0])
    parser.add_argument("--drs", nargs="+", default=['-9999'])
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--learning_starts", type=int, default=10000)
    parser.add_argument("--total_timesteps", type=int, default=3000000)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--pause", type=float, default=0.1)
    parser.add_argument("--dry", action='store_true')

    args, extra_args = parser.parse_known_args()
    run(**args.__dict__)
