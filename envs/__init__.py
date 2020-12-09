from gym.envs.registration import register

register(
    id='rooms-v0',
    entry_point='src.envs.rooms.rooms:RoomsEnv',
    max_episode_steps=100000,
    kwargs={}
)

