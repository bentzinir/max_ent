import torch as th
from torch.nn import functional as F
from stable_baselines3.common import logger


def action_probs(obs, next_obs, action_module, cat_dim=1):
    x = th.cat((obs, next_obs), dim=cat_dim).float()
    action_model_logits = action_module(x)
    return th.nn.Softmax(dim=1)(action_model_logits)


def active_mask(actions, action_model_probs, threshold):
    n_actions = action_model_probs.shape[1]
    if threshold is None:
        a_mask = F.one_hot(th.squeeze(actions), n_actions).float()
        pa_a = th.sum(a_mask * action_model_probs, dim=1, keepdim=True)
        # use relative threshold : thresh = p(a|s,s')
        threshold = pa_a.repeat(1, n_actions)
    active_actions = (action_model_probs >= threshold).float()
    if isinstance(threshold, float):
        mean_th = threshold
    else:
        mean_th = threshold.mean().item()
    logger.record("action model/threshold", mean_th, exclude="tensorboard")
    return active_actions


def discrete_min_red(obs, next_obs, actions, pi, pi_0, dones, method, importance_sampling, absolute_threshold, delta, cat_dim, action_module):
    '''
    :param obs: B x C x H x W
    :param next_obs: B x C x H x W
    :param actions: B x 1
    :param pi: B X |A|
    :param pi_0: B X |A|
    :param dones: B X 1
    :param method: str
    :param importance_sampling: bool
    :param absolute_threshold: bool
    :param cat_dim: int
    :param action_module:
    :return:
    '''

    n_actions = pi.shape[1]
    logger.record("action model/method", method, exclude="tensorboard")
    logger.record("action model/a_hist", th.histc(actions.float(), bins=n_actions).tolist())

    if method == 'Nill':
        return th.zeros_like(actions, dtype=th.float)

    action_model_probs = action_probs(obs, next_obs, action_module, cat_dim)
    a_mask = F.one_hot(th.squeeze(actions), n_actions).float()
    a_prime_mask = 1 - a_mask
    pi_a = th.sum(a_mask * pi, dim=1, keepdim=True)
    pa_a = th.sum(a_mask * action_model_probs, dim=1, keepdim=True)
    if absolute_threshold:
        thresh = delta * th.ones_like(action_model_probs)
    else:
        thresh = pa_a.repeat(1, n_actions)
    logger.record("action model/threshold", thresh.mean().item())
    active_actions = (action_model_probs >= thresh).float()
    pi_a_prime = th.sum(active_actions * a_prime_mask * pi, dim=1, keepdim=True)
    n_primes = th.mean(th.sum(active_actions * a_prime_mask, dim=1))
    logger.record("action model/n_primes", n_primes.item(), exclude="tensorboard")
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
            g = g * (pi_a / pi_0)
        # Mask with done
        g = g * (1 - dones)
        logger.record("action model/g", g.mean().item())
    return g, n_primes
