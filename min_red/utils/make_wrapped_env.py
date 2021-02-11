from typing import Any, Dict, Optional, Type, Union
from stable_baselines3.common.env_util import make_vec_env
import gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from min_red.utils.macro_action_wrapper import MacroActionWrapper
from min_red.utils.delayed_reward_wrapper import DelayedRewardWrapper
from min_red.utils.accumulated_reward_wrapper import AccumulatedRewardWrapper
from min_red.utils.wandb_wrapper import WandbWrapper


def make_wrapped_env(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[DummyVecEnv, SubprocVecEnv]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:

    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def macro_action_wrapper(env: gym.Env) -> gym.Env:
        if wrapper_kwargs.get('macro_length', 1) > 1:
            env = MacroActionWrapper(env, **wrapper_kwargs)

        env = DelayedRewardWrapper(env, dt=int(wrapper_kwargs.get('dt', 0)))

        env = AccumulatedRewardWrapper(env, dr=float(wrapper_kwargs.get('dr', -9999)))

        wandb_log_interval = wrapper_kwargs.get('wandb_log_interval', 0)
        if wandb_log_interval:
            env = WandbWrapper(env, wandb_log_interval=wandb_log_interval)
        return env

    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=macro_action_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )
