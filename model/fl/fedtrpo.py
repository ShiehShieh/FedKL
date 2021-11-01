from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
from tqdm import tqdm

import random
import numpy as np

import model.fl.fedbase as fedbase_lib


class FedTRPO(fedbase_lib.FederatedBase):

  def __init__(self, clients_per_round, num_rounds, num_iter,
               timestep_per_batch, max_steps, eval_every,
               drop_percent, verbose=False, svf_n_timestep=1e6, **kwargs):
    super(FedTRPO, self).__init__(
        clients_per_round, num_rounds, num_iter, timestep_per_batch,
        max_steps, eval_every,
        drop_percent)
    self.verbose = verbose
    self.svf_n_timestep = svf_n_timestep

  def train(self):
    logging.error('Training with {} workers per round ---'.format(self.clients_per_round))
    outer_loop = tqdm(
        total=self.num_rounds, desc='Round', position=0)
    for i in range(self.num_rounds):
      # test model
      if i % self.eval_every == 0:
          stats = self.test()  # have distributed the latest model.
          rewards = stats[2]
          outer_loop.write('At round {} expected future discounted reward: {}'.format(i, np.mean(rewards)))

      indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
      np.random.seed(i)
      cpr = self.clients_per_round
      if cpr > len(selected_clients):
        cpr = len(selected_clients)
      active_clients = np.random.choice(selected_clients, round(cpr * (1 - self.drop_percent)), replace=False)

      cws = []  # buffer for receiving client solutions

      # communicate the latest model
      inner_loop = tqdm(
          total=len(active_clients), desc='Client', position=1)
      self.distribute(active_clients)
      for idx, c in enumerate(active_clients):  # simply drop the slow devices
        # Sync local (global) params to local old policy before training.
        # c.sync_old_policy()
        # Enable svf so as to calculate norm constraint.
        c.enable_svf(self.svf_n_timestep)
        # Sequentially run train each client. Notice that, we do not sync
        # old policy before each local fit, but before round.
        c.experiment(num_iter=self.num_iter,
                     timestep_per_batch=self.timestep_per_batch,
                     callback_before_fit=[c.sync_old_policy],
                     logger=inner_loop.write if self.verbose else None)

        # gather weights from client
        cws.append((c.get_client_weight(), c.get_params()))

        # track communication cost
        # self.metrics.update(rnd=i, cid=c.id, stats=stats)
        inner_loop.update()

      # update models
      self.global_weights = self.aggregate(cws)

      outer_loop.update()

    # final test model
    stats = self.test()
    rewards = stats[2]
    # self.metrics.accuracies.append(stats)
    # tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
    logging.error('At round {} total reward received: {}'.format(self.num_rounds, np.mean(rewards)))
