import torch
import torch.nn as nn
from typing import Tuple, Any


class MLP(nn.Module):
    def __init__(self,
                 layers: Tuple[nn.Module, ...],
                 layer_dims: Tuple[int, ...],
                 init_funcs: Tuple[Any, ...] = None,
                 batch_normalization: bool = False):
        """
        Implements a generic MLP module

        :param layers: a tuple containing the type of layer.
        :param layer_dims: a tuple containing the dimensions of each of the layers.
                           layer_dims[0] is the input dimension
        :param init_funcs: a tuple containing the init functions for each of the layers.
        :param batch_normalization: set to true to use batch normalization
        """
        super(MLP, self).__init__()
        self.layer_dims = layer_dims
        self.hidden = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.batch_normalization = batch_normalization
        self.n_layers = len(layers)

        for k in range(len(layers)):
            layer = layers[k](self.layer_dims[k], self.layer_dims[k + 1])
            if init_funcs is not None:
                layer = init_funcs[k](layer)
            self.hidden.append(layer)
            if batch_normalization:
                batch_norm_layer = torch.nn.BatchNorm1d(self.layer_dims[k + 1])
                self.batch_norm_layers.append(batch_norm_layer)

    def forward(self, x, activation=None):
        for k in range(self.n_layers):
            x = self.hidden[k](x)
            if self.batch_normalization:
                x = self.batch_norm_layers[k](x)
            if k < len(self.hidden) - 1:
                x = torch.tanh(x)
            elif activation:
                x = activation(x)
        return x


if __name__ == '__main__':
    n_layers = 3
    input_dim = 12
    batch_size = 256
    layers = (nn.Linear,) * n_layers
    layer_dims = (input_dim, input_dim * 8, input_dim * 16, input_dim * 32)
    network = MLP(layers=layers,
                  layer_dims=layer_dims,
                  batch_normalization=True)

    x = torch.rand(batch_size, input_dim)
    network.train()
    out_train = network(x)
    network.eval()
    out_eval = network(x)

