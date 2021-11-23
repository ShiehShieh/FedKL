from __future__ import absolute_import, division, print_function

from absl import app, flags, logging

import sys
import csv
import random
import numpy as np

from mujoco_py.builder import MujocoException


class FederatedBase(object):

  def __init__(self, clients_per_round, num_rounds, num_iter,
               timestep_per_batch, max_steps, eval_every, drop_percent,
               retry_min=-sys.float_info.max, reward_history_fn=''):
    self.clients = []
    self.clients_per_round = clients_per_round
    self.num_rounds = num_rounds
    self.num_iter = num_iter
    self.timestep_per_batch = timestep_per_batch
    self.max_steps = max_steps
    self.eval_every = eval_every
    self.drop_percent = drop_percent
    self.global_weights = None
    self.retry_min = retry_min
    self.num_retry = 0
    self.reward_history_fn = reward_history_fn

  def register(self, client):
    self.clients.append(client)
    if self.global_weights is None:
      self.global_weights = client.get_params()

  def distribute(self, clients):
    for client in clients:
      client.set_params(self.global_weights)

  def select_clients(self, round_id, num_clients):
    num_clients = min(num_clients, len(self.clients))
    # make sure for each comparison, we are selecting the same clients each round
    np.random.seed(round_id)
    indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
    return indices, [self.clients[i] for i in indices]

  def aggregate(self, cws):
    # Simple averaging.
    total_weight = 0.0
    for (w, ws) in cws:  # w is the number of local samples
      total_weight += w
    averaged_ws = [0] * len(cws[0][1])
    for (w, ws) in cws:  # w is the number of local samples
      for i, v in enumerate(ws):
        averaged_ws[i] += (w / total_weight) * v.astype(np.float64)
    return averaged_ws

  def train(self):
    raise NotImplementedError

  def test(self, clients=None):
    self.distribute(self.clients)
    rewards = []
    if clients is None:
      clients = self.clients
    for c in clients:
      r = c.test()
      rewards.append(r)
    ids = [c.cid for c in self.clients]
    groups = [c.group for c in self.clients]
    return ids, groups, rewards

  def retry(self, fs, lamb, max_retry=100, logger=None, retry_min=None):
    """
    Retry the experiment when the local objective diverged. We're studying the
    effect of system heterogeneity and statistical heterogeneity, so we don't
    want to be borthered by local divergence. Here, we assume that we can always
    converge the local objective.
    """
    if retry_min is None:
      retry_min = self.retry_min
    i = -1
    r = retry_min
    while r <= retry_min:
      for f in fs:
        f()
      try:
        i += 1
        r = lamb()
      except MujocoException as e:
        if logger:
          logger('%s' % e)
        if i >= max_retry:
          raise e
      except Exception as e:
        if logger:
          logger('%s' % e)
        if i >= max_retry:
          raise e
      finally:
        if i >= max_retry:
          break
    self.num_retry += i
    return r

  def get_num_retry(self):
    return self.num_retry

  def log_csv(self, reward_history):
    if len(self.reward_history_fn) == 0:
      raise NotImplementedError('no reward_history_fn provided')
    with open(self.reward_history_fn, 'w', newline='') as csvfile:
      w = csv.writer(csvfile, delimiter=',',
                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
      w.writerows(reward_history)
