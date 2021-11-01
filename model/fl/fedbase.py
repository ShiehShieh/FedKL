from __future__ import absolute_import, division, print_function

from absl import app, flags, logging

# from multiprocessing.dummy import Pool as ThreadPool

import random
import numpy as np


class FederatedBase(object):

  def __init__(self, clients_per_round, num_rounds, num_iter,
               timestep_per_batch, max_steps, eval_every, drop_percent):
    self.clients = []
    self.clients_per_round = clients_per_round
    self.num_rounds = num_rounds
    self.num_iter = num_iter
    self.timestep_per_batch = timestep_per_batch
    self.max_steps = max_steps
    self.eval_every = eval_every
    self.drop_percent = drop_percent
    self.global_weights = None

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

  def test(self):
    self.distribute(self.clients)
    rewards = []
    # pool = ThreadPool(len(self.clients))
    # rewards = pool.map(lambda c: c.test(), self.clients)
    for c in self.clients:
      r = c.test()
      rewards.append(r)
    ids = [c.cid for c in self.clients]
    groups = [c.group for c in self.clients]
    return ids, groups, rewards
