import torch as th
from stable_baselines3.common import logger


def retrieve_logp(x, y, module):
    # 1.1 calculate Gaussian moments by propagating x in module
    mu, log_std, _ = module.get_action_dist_params(x)
    # 1.2 update probability distribution with calculated mu, log_std
    module.action_dist.proba_distribution(mu, log_std)
    return module.action_dist.log_prob(y)


def stochastic_min_red(obs, next_obs, action, actor, action_module, cat_dim, logp_fresh, logp0_replayed=None, eps=1e-4, importance_sampling=False):
    # 1. P(a|s,s')
    x = th.cat((obs, next_obs), dim=cat_dim).float()
    action_model_logp = retrieve_logp(x, action, action_module)

    # 2. Get the probability of replayed action in current policy
    logp_replayed = retrieve_logp(obs, action, actor)

    if importance_sampling:
        w = th.exp(logp_replayed.view(-1, 1) - logp0_replayed)
        action_model_logp = action_model_logp.view(-1, 1) * th.clamp(w, 1e-4, 1e1)
        logger.record("train/is_weights", w.mean().item(), exclude="tensorboard")

    return action_model_logp - logp_fresh

