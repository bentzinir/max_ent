import torch as th
from stable_baselines3.common import logger


def stochastic_min_red(obs, next_obs, action, actor, action_module, cat_dim, logp0=None, eps=1e-4, importance_sampling=False):
    # Entropy & Action Uniqueness regularization
    # 1. build s,s'=f(s,a) distribution function
    x = th.cat((obs, next_obs), dim=cat_dim).float()
    # 1.1 calculate mu, sigma of the Gaussian action model
    mu, log_std, _ = action_module.get_action_dist_params(x)
    # 1.2 update probability distribution with calculated mu, log_std
    action_module.action_dist.proba_distribution(mu, log_std)
    action_model_logp = action_module.action_dist.log_prob(action)

    # # 3. get the probability of replayed action from current policy
    mu, log_std, _ = actor.get_action_dist_params(obs)
    # # 3.1 update probability distribution with calculated mu, log_std
    actor.action_dist.proba_distribution(mu, log_std)
    log_p = actor.action_dist.log_prob(action)

    g = action_model_logp - log_p
    if importance_sampling:
        w = th.exp(log_p - logp0.view(log_p.shape))
        g = g * th.clamp(w, 1e-4, 4)
        logger.record("train/is_weights", w.mean().item(), exclude="tensorboard")
    return g
