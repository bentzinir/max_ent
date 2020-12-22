import gym
import torch
from torch import nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space


class CustomCnn(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCnn, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]
        input_dim = observation_space.shape[1]
        kernel_sizes = [max(input_dim // 8, 1), max(input_dim // 16, 1), max(input_dim // 21, 1)]
        strides = [max(input_dim // 16, 1), max(input_dim // 32, 1), 1]
        filters = [256 // kernel_sizes[0], 256 // kernel_sizes[1], 256 // kernel_sizes[2]]

        print('Creating custom CNN arch with the following parameters:')
        print(f'kernel sizes: {kernel_sizes}')
        print(f'strides: {strides}')
        print(f'filters: {filters}')
        print(f'feature dimension: {features_dim}')
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, filters[0], kernel_size=kernel_sizes[0], stride=strides[0], padding=0),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[1], kernel_size=kernel_sizes[1], stride=strides[1], padding=0),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[2], kernel_size=kernel_sizes[2], stride=strides[2], padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))