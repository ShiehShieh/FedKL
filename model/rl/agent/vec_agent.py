from __future__ import absolute_import, division, print_function

import numpy as np


class VecAgent(object):
  def __init__(self, agents):
    self.agents = agents
    self.num_agent = len(agents)

  def fit(self, steps_list, indices=None, logger=None):
    assert len(steps_list) == self.num_agent or self.num_agent == 1, 'mismatched steps dim and # agent: want: %s, got: %s' % (self.num_agent, len(steps_list))
    if indices is None:
      indices = range(self.num_agent)
    for i in indices:
      steps = steps_list[i]
      agent = self.agents[i]
      agent.critic.fit(steps['observations'], steps['dfr'])
      agent.policy.fit(steps, logger)

  def value(self, observations):
    assert len(observations) == self.num_agent or self.num_agent == 1, 'mismatched observation dim and # agent'
    if self.num_agent == 1:
      return self.agents[0].value(observations)
    out = []
    for i in range(self.num_agent):
      obs = observations[i]
      agent = self.agents[i]
      out.extend(agent.value([obs]))
    return out

  def act(self, observations):
    assert len(observations) == self.num_agent or self.num_agent == 1, 'mismatched observation dim and # agent'
    if self.num_agent == 1:
      return self.agents[0].act(observations)
    actions, probses = [], []
    for i in range(self.num_agent):
      obs = observations[i]
      agent = self.agents[i]
      acts, probs = agent.act([obs])
      actions.extend(acts)
      probses.extend(probs)
    return actions, probses

  def get_agent(self, i):
    return self.agents[i]

  def get_value_func(self, i):
    return self.agents[i].value
