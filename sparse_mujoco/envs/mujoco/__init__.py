from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from sparse_mujoco.envs.mujoco.ant import AntEnv
from sparse_mujoco.envs.mujoco.half_cheetah import HalfCheetahEnv
from sparse_mujoco.envs.mujoco.hopper import HopperEnv
from sparse_mujoco.envs.mujoco.walker2d import Walker2dEnv
from sparse_mujoco.envs.mujoco.humanoid import HumanoidEnv
from sparse_mujoco.envs.mujoco.humanoidstandup import HumanoidStandupEnv
