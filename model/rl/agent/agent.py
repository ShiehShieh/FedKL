from __future__ import absolute_import, division, print_function

import numpy as np

import random


class Agent(object):
  def __init__(self, agent_id,
               policy,
               critic=None,
               init_exp=0.5,
               final_exp=0.0,
               anneal_steps=500,
               expose_critic=False):
    self.policy = policy
    self.critic = critic

    self.epsilon               = init_exp
    self.init_exp              = init_exp
    self.final_exp             = final_exp
    self.anneal_steps          = anneal_steps
    self.expose_critic         = expose_critic

    self.num_policy_params = len(self.policy.get_params())

  def anneal_exploration(self, global_step):
    ratio = max((self.anneal_steps - global_step) / float(self.anneal_steps), 0)
    self.epsilon = (self.init_exp - self.final_exp) * ratio + self.final_exp

  def fit(self, steps, logger=None):
    if self.critic is not None:
      hist = self.critic.fit(steps['observations'], steps['dfr'])
    return self.policy.fit(steps, logger)

  def value(self, observations):
    if self.critic is None:
      return [0] * len(observations)
    return self.critic.predict(observations)

  def epsilon_greedy(self, observations):
    action, probs = self.act(observations)
    # epsilon-greedy exploration strategy
    if random.random() < self.epsilon:
      if len(action.shape) == 0:
        return random.randint(
            0, self.policy.num_actions - 1, size=action.shape[0]), probs
      return np.random.rand(*action.shape), probs
    else:
      return action, probs

  def act(self, observations):
    return self.policy.act(observations)

  def set_params(self, model_params):
    if self.expose_critic:
      p = model_params[:self.num_policy_params]
      c = model_params[self.num_policy_params:]
      return self.policy.set_params(p), self.critic.set_params(c)
    return self.policy.set_params(model_params)

  def get_params(self):
    if self.expose_critic:
      p = self.policy.get_params()
      c = self.critic.get_params()
      return p + c
    return self.policy.get_params()

  def reset_num_timestep_seen(self):
    return self.policy.reset_num_timestep_seen()

  def get_num_timestep_seen(self):
    return self.policy.get_num_timestep_seen()

  def set_nm_penalty_coefficient(self, v):
    self.policy.set_nm_penalty_coefficient(v)

  def adapt_nm_target(self, num):
    self.policy.adapt_nm_target(num)

  def reset_optimizer(self):
    return self.policy.reset_optimizer()

  def sync_optimizer(self):
    return self.policy.sync_optimizer()

  def sync_old_policy(self):
    return self.policy.sync_old_policy()

  def sync_anchor_policy(self):
    return self.policy.sync_anchor_policy()

  def sync_backup_policy(self):
    return self.policy.sync_backup_policy()

  def stat(self):
    return self.policy.stat()
