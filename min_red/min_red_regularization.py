import torch as th
from torch.nn import functional as F
from stable_baselines3.common import logger
import numpy as np


def min_red_th(obs, next_obs, actions, pi, method, importance_sampling, absolute_threshold, cat_dim, action_module):
    '''
    :param obs: B x C x H x W
    :param next_obs: B x C x H x W
    :param actions: B x 1
    :param pi: B X |A|
    :param method: str
    :param importance_sampling: bool
    :param absolute_threshold: bool
    :param cat_dim: int
    :param action_module:
    :return:
    '''
    if method == 'Nill':
        return th.zeros_like(actions, dtype=th.float)

    n_actions = pi.shape[1]
    x = th.cat((obs, next_obs), dim=cat_dim).float()
    action_model_logits = action_module(x)
    action_model_probs = th.nn.Softmax(dim=1)(action_model_logits)
    a_mask = F.one_hot(th.squeeze(actions), n_actions).float()
    a_prime_mask = 1 - a_mask
    pi_a = th.sum(a_mask * pi, dim=1, keepdim=True)
    pa_a = th.sum(a_mask * action_model_probs, dim=1, keepdim=True)
    if absolute_threshold:
        thresh = 1e-2
    else:
        thresh = pa_a.repeat(1, n_actions)
    active_actions = (action_model_probs >= thresh).float()
    pi_a_prime = th.sum(active_actions * a_prime_mask * pi, dim=1, keepdim=True)
    n_primes = th.mean(th.sum(active_actions * a_prime_mask, dim=1))
    logger.record("action model/n_primes", n_primes.item(), exclude="tensorboard")
    logger.record("action model/method", method, exclude="tensorboard")
    with th.no_grad():
        eps = 1e-4
        if method == 'Nill':
            g = th.zeros_like(pi_a)
        elif method == 'action':
            g = - th.log(pi_a + eps)
        elif method == 'eta':
            g = - th.log(pi_a + pi_a_prime + eps)
        elif method == 'stochastic':
            g = th.log(pa_a + eps) - th.log(pi_a + eps)
        else:
            raise ValueError
        if importance_sampling:
            g = g * (pi_a / pi)
        logger.record("train/g", g.mean().item())
    return g


def min_red_np(obs, next_obs, actions, pi, method, importance_sampling, absolute_threshold, cat_dim, action_module):
    '''
    :param obs: B x C x H x W
    :param next_obs: B x C x H x W
    :param actions: B x 1
    :param pi: B X |A|
    :param method: str
    :param importance_sampling: bool
    :param absolute_threshold: bool
    :param cat_dim: int
    :param action_module:
    :return:
    '''
    n_actions = pi.shape[1]
    x = np.concatenate([obs, next_obs], axis=1).astype(np.float32)
    action_model_logits = action_module(x)
    action_model_probs = th.nn.Softmax(dim=1)(action_model_logits)
    a_mask = F.one_hot(th.squeeze(actions), n_actions).float()
    a_prime_mask = 1 - a_mask
    pi_a = th.sum(a_mask * pi, dim=1, keepdim=True)
    pa_a = th.sum(a_mask * action_model_probs, dim=1, keepdim=True)
    if absolute_threshold:
        thresh = 1e-2
    else:
        thresh = pa_a.repeat(1, n_actions)
    active_actions = (action_model_probs >= thresh).float()
    pi_a_prime = th.sum(active_actions * a_prime_mask * pi, dim=1, keepdim=True)
    n_primes = th.mean(th.sum(active_actions * a_prime_mask, dim=1))
    logger.record("action model/n_primes", n_primes.item(), exclude="tensorboard")
    logger.record("action model/method", method, exclude="tensorboard")
    with th.no_grad():
        eps = 1e-4
        if method == 'none':
            g = 0.0
        elif method == 'action':
            g = - th.log(pi_a + eps)
        elif method == 'eta':
            g = - th.log(pi_a + pi_a_prime + eps)
            g = g
        elif method == 'stochastic':
            g = th.log(pa_a + eps) - th.log(pi_a + eps)
        else:
            raise ValueError
        if importance_sampling:
            g = g * (pi_a / pi)
    return g
