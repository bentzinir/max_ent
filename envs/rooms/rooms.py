import numpy as np
from gym import core, spaces
import random
import cv2
from tqdm import tqdm
import time


class RoomsEnv(core.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rows=16, cols=16, empty=False, random_walls=False, discrete=True,
                 spatial=True, n_redundancies=1, max_repeats=1, goal=None, state=None,
                 goal_in_state=True, max_steps=None,
                 goal_only_visible_in_room=False, seed=None,
                 fixed_reset=False, vert_wind=(0, 0), horz_wind=(0, 0)):
        '''
        vert_wind = (up, down)
        horz_wind = (right, left)
        '''
        self.rows, self.cols = rows, cols
        if max_steps is None:
            repeats = 3 if empty else 7
            self.max_steps = (rows + cols) * repeats
        else:
            self.max_steps = max_steps

        self.goal_in_state = goal_in_state
        self.goal_only_visible_in_room = goal_only_visible_in_room

        self.vert_wind = np.array(vert_wind)
        self.horz_wind = np.array(horz_wind)

        self.n_redundancies = n_redundancies
        self.max_repeats = max_repeats
        self.spatial = spatial
        self.discrete = discrete
        self.scale = np.maximum(rows, cols)
        self.im_size = 42
        # Action space
        if self.discrete:
            self.action_space = spaces.Discrete(3 + n_redundancies)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2 + n_redundancies, 1))
        # Observation Space
        if spatial:
            n_channels = 2 + goal_in_state
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.im_size, self.im_size, n_channels), dtype=np.uint8)
        else:
            n_channels = 2 + goal_in_state * 2
            self.observation_space = spaces.Box(low=0, high=1, shape=(n_channels,), dtype=np.float32)

        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1))] + [np.array((0, 1))] * n_redundancies
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        self.map, self.seed = self._randomize_walls(random=random_walls, empty=empty)
        self.goal_cell, self.goal = self._random_from_map(goal)
        self.state_cell, self.state = self._random_from_map(state)

        self.fixed_reset = fixed_reset
        if fixed_reset:
            self.reset_state_cell, self.reset_state = self.state_cell.copy(), self.state.copy()
        else:
            self.reset_state_cell, self.reset_state = None, None

        self.tot_reward = 0
        self.viewer = None

    def reset(self):
        if self.fixed_reset:
            self.state_cell, self.state = self.reset_state_cell, self.reset_state
        else:
            self.state_cell, self.state = self._random_from_map(None)

        self.nsteps = 0
        self.tot_reward = 0

        obs = self._obs_from_state(self.spatial)
        return obs

    def step(self, action: int):
        # actions: 0 = up, 1 = down, 2 = left, 3:end = right
        for _ in range(random.randint(1, self.max_repeats)):
            self._move(action)
            wind_up = np.random.choice([-1, 0, 1], p=[1 - self.vert_wind.sum(), self.vert_wind[0], self.vert_wind[1]])
            wind_right = np.random.choice([-1, 2, 3], p=[1 - self.horz_wind.sum(), self.horz_wind[1], self.horz_wind[0]])
            if wind_up >= 0:
                self._move(wind_up, discrete=True)
            if wind_right >= 0:
                self._move(wind_right, discrete=True)

            done = np.all(self.state_cell == self.goal_cell)
            obs = self._obs_from_state(self.spatial)
            r = float(done)

            if self.nsteps >= self.max_steps:
                done = True

            self.tot_reward += r
            self.nsteps += 1
            info = dict()

            if done:
                info['episode'] = {'r': self.tot_reward, 'l': self.nsteps}
                break

        return obs, r, done, info

    def _next_cell_discrete(self, action):
        return self.state_cell + self.directions[action]

    def _next_cell_continuous(self, action):
        xy = action[:2]
        angles = action[2:]
        # for angle in angles:
        #     theta = angle[0] * np.pi
        #     R = np.array([[np.cos(theta), -np.sin(theta)],
        #                   [np.sin(theta), np.cos(theta)]])
        #     xy = R @ xy
        xy = self._star_deform(xy)

        next_cell = np.round(self.state_cell + xy.squeeze()).astype(np.int)
        return next_cell

    def _star_deform(self, xy):
        x, y = xy
        dx = np.abs(y) * x**2 * np.sign(-x)
        dy = np.abs(x) * y**1 * np.sign(-y)
        return np.array([x+dx, y+dy])

    def _move(self, action, discrete=None):
        if self.discrete or discrete:
            next_cell = self._next_cell_discrete(action)
        else:
            next_cell = self._next_cell_continuous(action)

        if self.map[next_cell[0], next_cell[1]] == 0:
            self.state_cell = next_cell
            self.state = np.zeros_like(self.map)
            self.state[(self.state_cell[0]):(self.state_cell[0] + 1),
                       (self.state_cell[1]):(self.state_cell[1] + 1)] = 1

    def _random_from_map(self, goal):

        if goal is None:
            cell = self.rng.choice(self.rows), self.rng.choice(self.cols)
        else:
            cell = tuple(goal)
        while self.map[cell[0], cell[1]] == 1:
            cell = self.rng.choice(self.rows), self.rng.choice(self.cols)
        map = np.zeros_like(self.map)
        for i in [0]:#[-1, 0, 1]:
            for j in [0]:#[-1, 0, 1]:
                map[(cell[0] + i):
                    (cell[0] + i + 1),
                    (cell[1] + j):
                    (cell[1] + j + 1)] = 1

        return np.array(cell), map

    def _obs_from_state(self, spatial):
        if spatial:
            im_list = [self.state, self.map]
            if self.goal_in_state:
                if self.goal_only_visible_in_room:
                    if self._which_room(self.state_cell) == self._which_room(self.goal_cell):
                        im_list.append(self.goal)
                    else:
                        im_list.append(np.zeros_like(self.map))
                else:
                    im_list.append(self.goal)
            im_stack = 70 * np.stack(im_list, axis=-1).astype(np.uint8)
            return cv2.resize(im_stack, dsize=(self.im_size, self.im_size), interpolation=cv2.INTER_AREA)
        else:
            obs = list(self.state_cell)
            if self.goal_in_state:
                if self.goal_only_visible_in_room:
                    if self._which_room(self.state_cell) == self._which_room(self.goal_cell):
                        obs += list(self.goal_cell)
                else:
                    obs += list(self.goal_cell)
            return np.array(obs) / self.scale

    def _which_room(self, cell):
        if cell[0] <= self.seed[0] and cell[1] <= self.seed[1]:
            return 0
        elif cell[0] <= self.seed[0] and cell[1] > self.seed[1]:
            return 1
        elif cell[0] > self.seed[0] and cell[1] <= self.seed[1]:
            return 2
        else:
            return 3

    def _randomize_walls(self, random=False, empty=False):
        map = np.zeros((self.rows, self.cols))

        map[0, :] = 1
        map[:, 0] = 1
        map[-1:, :] = 1
        map[:, -1:] = 1

        if empty:
            return map, 0

        if random:
            seed = (self.rng.randint(2, self.rows - 2), self.rng.randint(2, self.cols - 2))
            doors = (self.rng.randint(1, seed[0]),
                     self.rng.randint(seed[0] + 1, self.rows - 1),
                     self.rng.randint(1, seed[1]),
                     self.rng.randint(seed[1] + 1, self.cols - 1))
        else:
            seed = (self.rows // 2, self.cols // 2)
            doors = (self.rows // 4, 3 * self.rows // 4, self.cols // 4, 3 * self.cols // 4)

        map[seed[0]:seed[0] + 1, :] = 1
        map[:, seed[1]:(seed[1] + 1)] = 1
        map[doors[0]:(doors[0]+1), seed[1]:(seed[1] + 1)] = 0
        map[doors[1]:(doors[1]+1), seed[1]:(seed[1] + 1)] = 0
        map[seed[0]:(seed[0] + 1), doors[2]:(doors[2]+1)] = 0
        map[seed[0]:(seed[0] + 1), doors[3]:(doors[3]+1)] = 0

        return map, seed

    def render(self, mode='human'):
        img = self._obs_from_state(True)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


if __name__ == '__main__':

    n_redundancies = 1
    discrete = False
    spatial = True
    max_repeats = 1
    room_size = 42
    up_wind = 0
    down_wind = 0.0
    right_wind = 0.0
    left_wind = 0

    env = RoomsEnv(rows=room_size, cols=room_size, discrete=discrete, spatial=spatial,
                   goal=[1, 1], state=[room_size - 2, room_size - 2],
                   fixed_reset=True, n_redundancies=n_redundancies, max_repeats=max_repeats,
                   horz_wind=(right_wind, left_wind), vert_wind=(up_wind, down_wind), empty=False, seed=0)

    obs = env.reset()
    for _ in tqdm(range(10000), desc='', leave=True):
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        time.sleep(0.03)
        if done:
            obs = env.reset()
