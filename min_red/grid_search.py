import os
envs = ['rooms-v0']
config_name = 'rooms'
alg = 'PPO'
methods = ['action', 'eta']
n_repeats = 2
abs_thresh = True
total_timesteps = 100000
log_interval = 1  # (episodes)
n_redundancies = 30
n_steps = 100

for trials in range(n_repeats):
    for env in envs:
        for method in methods:
            cmd_line = f"python -m min_red.train " \
                       f" --f min_red/config/{config_name} " \
                       f" --algorithm_type {alg} " \
                       f" --algorithm.learn.log_interval {log_interval} " \
                       f"--algorithm.policy.absolute_threshold {abs_thresh} " \
                       f"--method {method} " \
                       f"--env_kwargs.n_redundancies {n_redundancies} " \
                       f"--algorithm.policy.n_steps {n_steps} " \
                       f" --wandb True & "
            print(cmd_line)
            os.system(cmd_line)
