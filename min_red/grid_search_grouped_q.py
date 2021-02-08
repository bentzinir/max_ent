import os
import time
import argparse


def run(env_id, delta, macro_length, n_repeats, wandb_log_interval, total_timesteps, pause, dry):

    methods = [
              'baseline',
              'group'
               ]

    log_interval = 100

    for trials in range(n_repeats):
        for method in methods:
            cmd_line = f"python -m min_red.train --f min_red/config/atari --algorithm_type GroupedQ " \
                       f" --wrapper_kwargs.wandb_log_interval {wandb_log_interval} " \
                       f" --method {method} " \
                       f" --total_timesteps {total_timesteps} " \
                       f" --algorithm.learn.log_interval {log_interval} " \
                       f" --algorithm.policy.threshold {delta}" \
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
    parser.add_argument("--wandb_log_interval", type=int, default=10000)
    parser.add_argument("--total_timesteps", type=int, default=10000000)
    parser.add_argument("--pause", type=float, default=0.1)
    parser.add_argument("--dry", action='store_true')

    args, extra_args = parser.parse_known_args()
    run(**args.__dict__)

    # envs = [
    #     "BreakoutNoFrameskip-v4",
    #     "MsPacmanNoFrameskip-v4",
    #     "RiverraidNoFrameskip-v4",
    #     "RobotankNoFrameskip-v4",
    #     "SeaquestNoFrameskip-v4",
    #     "DemonAttackNoFrameskip-v4"
    #     "AirRaidNoFrameskip-v4",   # hard
    #     "PrivateEyeNoFrameskip-v4",   # hard
    #     "QbertNoFrameskip-v4",   # hard
    #            - AsteroidsNoFrameskip-v4
    #            - AmidarNoFrameskip-v4
    #            - BowlingNoFrameskip-v4
    #            - FrostbiteNoFrameskip-v4
    #            - GravitarNoFrameskip-v4
    #            - KrullNoFrameskip-v4
    #            - NameThisGameNoFrameskip-v4

    #            - SpaceInvadersNoFrameskip-v4
    #            - StarGunnerNoFrameskip-v4
    #            - TennisNoFrameskip-v4
    #            - TimePilotNoFrameskip-v4
    #            - TutankhamNoFrameskip-v4
    #            - WizardOfWorNoFrameskip-v4
    #     "MontezumaRevengeNoFrameskip-v4",  # hard
    #            - AlienNoFrameskip-v4  # hard
    # ]
