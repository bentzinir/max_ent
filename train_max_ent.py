import gym
import envs
import numpy as np
import time
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv
from action_model_trainer import ActionModelTrainer
import torch
import GPUtil


def eval_policy(env, model, steps=1000, desc=''):
    obs = env.reset()
    traj_rewards = [0]
    for _ in tqdm(range(steps), desc=desc, leave=True):
        action, _state = model.predict(obs, deterministic=False)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        env.render()
        time.sleep(0.03)
        traj_rewards[-1] += reward
        if done:
            obs = env.reset()
            traj_rewards.append(0)
    return np.mean(traj_rewards)


if torch.cuda.is_available():
    deviceIds = GPUtil.getFirstAvailable(order='memory', maxLoad=0.8, maxMemory=0.8)
    device = torch.device(f'cuda:{deviceIds[0]}')
else:
    device = torch.device('cpu')

lr = 2e-4
gamma = 0.9
buffer_size = 50000
batch_size = 64
learning_starts = 50000
total_timesteps = 250000
ent_coef = 0.001
exploration_final_rate = .1
# Regularization types:
# 1. none: g = 0
# 2. action: g = -log_pi
# 3. next_det: g = -log (pi_a + eta)
# 4. next_abs: g = |pa/p_i - 1|
# 5. next_log: g = log(pa/pi)
method = 'next_log'
discrete = True
n_redundancies = 4
max_repeats = 3
room_size = 10
up_wind = 0
down_wind = 0.2
right_wind = 0.2
left_wind = 0

env = DummyVecEnv([lambda: gym.make('rooms-v0', rows=room_size, cols=room_size, discrete=discrete,
                                    goal=[1, 1], state=[room_size - 2, room_size - 2],
                                    fixed_reset=True, n_redundancies=n_redundancies, max_repeats=max_repeats,
                                    horz_wind=(right_wind, left_wind), vert_wind=(up_wind, down_wind),
                                    empty=False, seed=0,)])

obs_shape = list(env.observation_space.shape)

if discrete:
    from max_ent_dqn import MaxEntDQN as Algorithm
    from stable_baselines3.dqn.policies import CnnPolicy as Model
    ssprime_shape = (2 * obs_shape[2], *obs_shape[:2])
    policy = 'CnnPolicy'
    cat_dim = 1
else:
    from max_ent_sac import MaxEntSAC as Algorithm
    from stable_baselines3.sac import MlpPolicy as Model
    # from continuous_model import ContModel as Model
    ssprime_shape = (2*obs_shape[0],)
    policy = 'MlpPolicy'
    cat_dim = 1

# create action model obs space by extending env's obs space
ssprime_obs_space = gym.spaces.Box(low=env.observation_space.low.min(),
                                   high=env.observation_space.high.max(),
                                   shape=ssprime_shape,
                                   dtype=env.observation_space.dtype)

action_model = Model(observation_space=ssprime_obs_space,
                     action_space=env.action_space,
                     lr_schedule=lambda x: lr)

action_trainer = ActionModelTrainer(action_model=action_model, cat_dim=cat_dim, discrete=discrete, lr=lr)
model = Algorithm(policy, env, verbose=1, gamma=gamma, buffer_size=buffer_size, learning_starts=learning_starts,
                  action_trainer=action_trainer, device=device, ent_coef=ent_coef, method=method,
                  batch_size=batch_size, exploration_final_eps=exploration_final_rate,
                  policy_kwargs={})
model.learn(total_timesteps=total_timesteps, log_interval=10)
model.save("rooms")

eval_res = eval_policy(env, model, desc='Evaluating model')
print(f'Eval Result = {eval_res}')