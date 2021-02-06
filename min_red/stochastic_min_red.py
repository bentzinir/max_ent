import torch as th
from stable_baselines3.common import logger


def retrieve_logp(x, y, module):
    # 1.1 calculate Gaussian moments by propagating x in module
    mu, log_std, _ = module.get_action_dist_params(x)
    # 1.2 update probability distribution with calculated mu, log_std
    module.action_dist.proba_distribution(mu, log_std)
    return module.action_dist.log_prob(y).view(-1, 1)


def stochastic_min_red(obs, next_obs, action, actor, action_module, cat_dim, logp0_replayed=None, importance_sampling=False):
    # 1. P(a|s,s')
    x = th.cat((obs, next_obs), dim=cat_dim).float()
    action_model_logp = retrieve_logp(x, action, action_module)
    return action_model_logp

