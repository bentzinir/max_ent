import os
import time

envs = ['Hopper-v2', 'HalfCheetah-v2', 'Ant-v2', 'Swimmer-v2', 'Walker2d-v2', 'Humanoid-v2']
methods = ['entropy', 'next-action', 'mixture', 'semi-mixture']
n_repeats = 2
ensemble_sizes = [3]
step_mixture = 'true'
total_timesteps = 3000000
buffer_size = 300000
log_interval = 1  # (episodes)
wandb_log_interval = 10000  # (timesteps)

# Debug
n_repeats = 2
buffer_size = 1000000
envs = ['Hopper-v2']
methods = ['mixture', 'semi-mixture', 'entropy']
# ensemble_sizes = [1, 2, 3]

for trials in range(n_repeats):
    for env in envs:
        for method in methods:
            for ens_size in ensemble_sizes:
                cmd_line = f"python -m mixture.train --f=mixture/config/mujoco " \
                           f" --wandb_log_interval {wandb_log_interval} " \
                           f" --total_timesteps {total_timesteps} " \
                           f" --algorithm.learn.log_interval {log_interval} " \
                           f" --step_mixture {step_mixture}" \
                           f" --ensemble_size {ens_size} " \
                           f" --algorithm.policy.buffer_size {buffer_size} " \
                           f" --method {method} " \
                           f" --env_id {env} & "
                print(cmd_line)
                os.system(cmd_line)
                time.sleep(0.1)