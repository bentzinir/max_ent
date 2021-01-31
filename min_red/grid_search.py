import os
import time

envs = [

    "BreakoutNoFrameskip-v4",
    # "MsPacmanNoFrameskip-v4",
    # "RiverraidNoFrameskip-v4",
    # "RobotankNoFrameskip-v4",
    # "SeaquestNoFrameskip-v4",

        # "AirRaidNoFrameskip-v4",   # hard
        # "PrivateEyeNoFrameskip-v4",   # hard
        # "QbertNoFrameskip-v4",   # hard
        # "MontezumaRevengeNoFrameskip-v4",  # hard
        #        - AlienNoFrameskip-v4  # hard
        #        - AmidarNoFrameskip-v4
        #        - AsteroidsNoFrameskip-v4
        #        - BowlingNoFrameskip-v4
        #        - DemonAttackNoFrameskip-v4
        #        - FrostbiteNoFrameskip-v4
        #        - GravitarNoFrameskip-v4
        #        - KrullNoFrameskip-v4
        #        - NameThisGameNoFrameskip-v4
        #        - RiverraidNoFrameskip-v4
        #        - RobotankNoFrameskip-v4
        #        - SeaquestNoFrameskip-v4
        #        - SpaceInvadersNoFrameskip-v4
        #        - StarGunnerNoFrameskip-v4
        #        - TennisNoFrameskip-v4
        #        - TimePilotNoFrameskip-v4
        #        - TutankhamNoFrameskip-v4
        #        - WizardOfWorNoFrameskip-v4
        ]

methods = [
          'baseline',
          'group'
           ]

n_repeats = 2
delta = 0.1
macro_length = 2
log_interval = 100
wandb_log_interval = 10000  # (timesteps)
total_timesteps = 3000000


for trials in range(n_repeats):
    for env in envs:
        for method in methods:
            cmd_line = f"python -m min_red.train --f min_red/config/atari --algorithm_type GroupedQ " \
                       f" --wandb_log_interval {wandb_log_interval} " \
                       f" --method {method} " \
                       f" --total_timesteps {total_timesteps} " \
                       f" --algorithm.learn.log_interval {log_interval} " \
                       f" --algorithm.policy.threshold {delta}" \
                       f" --macro_length {macro_length} " \
                       f" --env_id {env} & "
            print(cmd_line)
            os.system(cmd_line)
            time.sleep(0.1)