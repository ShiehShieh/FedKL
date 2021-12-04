from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
from tqdm import tqdm

import sys
import csv
import random
import numpy as np

from mujoco_py.builder import MujocoException

import model.rl.agent.vec_agent as vec_agent_lib
import model.utils.vectorization as vectorization_lib


class FederatedBase(object):

  def __init__(self, clients_per_round, num_rounds, num_iter,
               timestep_per_batch, max_steps, eval_every, drop_percent,
               retry_min=-sys.float_info.max, universial_client=None,
               reward_history_fn=''):
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
    self.universial_client = universial_client

  def register_universal_client(self, universial_client):
    self.universial_client = universial_client

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

  def _inner_sequential_loop(self, i_iter, active_clients, retry_min):
    raise NotImplementedError

  def _inner_vectorized_loop(self, i_iter, active_clients, retry_min):
    raise NotImplementedError

  def train(self):
    logging.error('Training with {} workers per round ---'.format(self.clients_per_round))
    retry_min = self.retry_min
    reward_history = []
    outer_loop = tqdm(
        total=self.num_rounds, desc='Round', position=0,
        dynamic_ncols=True)
    for i in range(self.num_rounds):
      # test model
      if i % self.eval_every == 0:
        if self.universial_client is not None:
          stats = self.universal_test()
        else:
          stats = self.test()  # have distributed the latest model.
        rewards = stats[2]
        retry_min = np.mean(rewards)
        reward_history.append(rewards)
        self.log_csv(reward_history)
        outer_loop.write(
            'At round {} expected future discounted reward: {}; # retry so far {}'.format(
                i, np.mean(rewards), self.get_num_retry()),
            file=sys.stderr)

      # uniform sampling
      indices, selected_clients = self.select_clients(
          i, num_clients=self.clients_per_round)
      np.random.seed(i)
      cpr = self.clients_per_round
      if cpr > len(selected_clients):
        cpr = len(selected_clients)
      active_clients = np.random.choice(selected_clients, round(cpr * (1 - self.drop_percent)), replace=False)

      # communicate the latest model
      self.distribute(active_clients)
      # buffer for receiving client solutions
      cws = []
      # Inner sequantial loop.
      if self.universial_client is not None:
        cws = self._inner_vectorized_loop(i, active_clients, retry_min)
      else:
        cws = self._inner_sequential_loop(i, active_clients, retry_min)

      # update models
      self.global_weights = self.aggregate(cws)

      outer_loop.update()

    # final test model
    if self.universial_client is not None:
      stats = self.universal_test()
    else:
      stats = self.test()  # have distributed the latest model.
    rewards = stats[2]
    reward_history.append(rewards)
    self.log_csv(reward_history)
    outer_loop.write(
        'At round {} total reward received: {}'.format(self.num_rounds, np.mean(rewards)),
        file=sys.stderr)
    return reward_history

  def test(self, clients=None):
    self.distribute(self.clients)
    rewards = []
    if clients is None:
      clients = self.clients
    for c in clients:
      r = self.retry(
          [],
          lambda: c.test(),
          max_retry=5,
          logger=None,
          retry_min=-sys.float_info.max,
      )
      rewards.append(r)
    ids = [c.cid for c in self.clients]
    groups = [c.group for c in self.clients]
    return ids, groups, rewards

  def universal_test(self):
    self.distribute(self.clients)
    agents = vec_agent_lib.VecAgent(
        [c.agent for c in self.clients])
    obfilts = vectorization_lib.VecCallable(
        [c.obfilt for c in self.clients])
    rewfilts = vectorization_lib.VecCallable(
        [c.rewfilt for c in self.clients])
    rewards = self.universial_client.test(agents, obfilts, rewfilts)
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
