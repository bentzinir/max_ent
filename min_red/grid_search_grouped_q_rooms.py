import os
import time
import argparse


def run(env_id, delta, macro_length, methods, n_redundancies, n_repeats, wandb_log_interval, total_timesteps, pause, dry):

    log_interval = 100
    learning_starts = 10000
    reg_starts = 20000
    for trials in range(n_repeats):
        for method in methods:
            for n_redun in n_redundancies:
                cmd_line = f"python -m min_red.train --f min_red/config/atari --algorithm_type GroupedQ " \
                           f" --wrapper_kwargs.wandb_log_interval {wandb_log_interval} " \
                           f" --method {method} " \
                           f" --env_kwargs.n_redundancies {n_redun} " \
                           f" --total_timesteps {total_timesteps} " \
                           f" --algorithm.learn.log_interval {log_interval} " \
                           f" --algorithm.policy.threshold {delta}" \
                           f" --algorithm.policy.learning_starts {learning_starts}"\
                           f" --algorithm.policy.regularization_starts {reg_starts}" \
                           f" --wrapper_kwargs.macro_length {macro_length} " \
                           f" --env_id {env_id} & "
                print(cmd_line)
                if not dry:
                    os.system(cmd_line)
                    time.sleep(pause)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="none")
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--macro_length", type=int, default=1)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--methods", nargs="+", default=['action', 'group'])
    parser.add_argument("--n_redundancies", nargs="+", default=[1])
    parser.add_argument("--wandb_log_interval", type=int, default=1000)
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--pause", type=float, default=0.1)
    parser.add_argument("--dry", action='store_true')

    args, extra_args = parser.parse_known_args()
    run(**args.__dict__)
