from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
from tqdm import tqdm

import sys
import random
import numpy as np

import model.fl.fedbase as fedbase_lib


class FedProx(fedbase_lib.FederatedBase):

  def __init__(self, clients_per_round, num_rounds, num_iter,
               timestep_per_batch, max_steps, eval_every,
               drop_percent, verbose=False, retry_min=-sys.float_info.max,
               reward_history_fn='', **kwargs):
    super(FedProx, self).__init__(
        clients_per_round, num_rounds, num_iter, timestep_per_batch,
        max_steps, eval_every, drop_percent, retry_min, reward_history_fn)
    self.verbose = verbose

  def train(self):
    logging.error('Training with {} workers per round ---'.format(self.clients_per_round))
    verbose = self.verbose
    retry_min = self.retry_min
    reward_history = []
    outer_loop = tqdm(
        total=self.num_rounds, desc='Round', position=0,
        dynamic_ncols=True)
    for i in range(self.num_rounds):
      # test model
      if i % self.eval_every == 0:
          stats = self.test()  # have distributed the latest model.
          rewards = stats[2]
          retry_min = np.mean(rewards)
          reward_history.append(rewards)
          self.log_csv(reward_history)
          outer_loop.write(
              'At round {} expected future discounted reward: {}; # retry so far {}'.format(
                  i, np.mean(rewards), self.get_num_retry()))

      indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
      np.random.seed(i)
      cpr = self.clients_per_round
      if cpr > len(selected_clients):
        cpr = len(selected_clients)
      active_clients = np.random.choice(selected_clients, round(cpr * (1 - self.drop_percent)), replace=False)

      cws = []  # buffer for receiving client solutions

      # communicate the latest model
      inner_loop = tqdm(
          total=len(active_clients), desc='Client', position=1,
          dynamic_ncols=True)
      self.distribute(active_clients)
      for idx, c in enumerate(active_clients):
        # Sequentially train each client.
        self.retry(
            [
                lambda: self.distribute([c]),
                lambda: c.reset_client_weight(),
                # sync local (global) params to local optimizer.
                lambda: c.sync_optimizer(),
                # sync local (global) params to local anchor.
                lambda: c.sync_anchor_policy(),
            ],
            lambda: c.experiment(
                num_iter=self.num_iter,
                timestep_per_batch=self.timestep_per_batch,
                callback_before_fit=[c.sync_old_policy],
                logger=inner_loop.write if verbose else None,
            ),
            max_retry=5 if i > 3 else 0,
            logger=inner_loop.write if verbose else None,
            retry_min=retry_min - np.abs(retry_min),
        )

        # gather weights from client
        cws.append((c.get_client_weight(), c.get_params()))

        # track communication cost
        # self.metrics.update(rnd=i, cid=c.cid, stats=stats)
        inner_loop.update()

      # update models
      self.global_weights = self.aggregate(cws)

      outer_loop.update()

    # final test model
    stats = self.test()
    rewards = stats[2]
    reward_history.append(rewards)
    self.log_csv(reward_history)
    outer_loop.write('At round {} total reward received: {}'.format(self.num_rounds, np.mean(rewards)))
    return reward_history
