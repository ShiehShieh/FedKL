from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
from tqdm import tqdm

import sys
import random
import numpy as np

import model.fl.fedbase as fedbase_lib


class FedTRPO(fedbase_lib.FederatedBase):

  def __init__(self, clients_per_round, num_rounds, num_iter,
               timestep_per_batch, max_steps, eval_every,
               drop_percent, verbose=False, svf_n_timestep=1e6,
               retry_min=-sys.float_info.max, **kwargs):
    super(FedTRPO, self).__init__(
        clients_per_round, num_rounds, num_iter, timestep_per_batch,
        max_steps, eval_every, drop_percent, retry_min)
    self.verbose = verbose
    self.svf_n_timestep = svf_n_timestep

  def get_state_visitation_frequency(self, active_clients, logger=None):
    svf_ms = []
    for idx, c in enumerate(active_clients):
      c.enable_svf(self.svf_n_timestep)
      d = c.get_state_visitation_frequency()
      svf_ms.append(d)
      if logger:
        logger('client id: %s, # state %s, l2 norm: %.10e' % (
            c.cid, len(d), np.linalg.norm(list(d.values()), ord=2)))
    full_keys = {}
    for svf_m in svf_ms:
      for k in svf_m.keys():
        if k in full_keys:
          continue
        full_keys[k] = len(full_keys)
    if logger:
      logger('# keys: %d' % (len(full_keys)))
    svfs = np.zeros(shape=(len(active_clients), len(full_keys)))
    for i, svf_m in enumerate(svf_ms):
      for k, v in svf_m.items():
        j = full_keys[k]
        svfs[i][j] += v
    # svfs = svfs / np.sum(svfs, axis=1)[:, np.newaxis]
    avg = np.mean(svfs, axis=0)
    norm_penalties = np.linalg.norm(avg - svfs, ord=2, axis=1)
    # np.sqrt(np.mean(np.square(np.linalg.norm(svfs, ord=2, axis=1))))
    if logger:
      logger('norm_penalties shape %s, l2 norm: %s' % (
          norm_penalties.shape, norm_penalties))
    return svfs, norm_penalties

  def train(self):
    verbose = self.verbose
    reward_history = []
    logging.error('Training with {} workers per round ---'.format(self.clients_per_round))
    outer_loop = tqdm(
        total=self.num_rounds, desc='Round', position=0,
        dynamic_ncols=True)
    for i in range(self.num_rounds):
      # test model
      if i % self.eval_every == 0:
          stats = self.test()  # have distributed the latest model.
          rewards = stats[2]
          reward_history.append(rewards)
          outer_loop.write(
              'At round {} expected future discounted reward: {}; # retry so far {}'.format(
                  i, np.mean(rewards), self.get_num_retry()))

      # uniform sampling
      indices, selected_clients = self.select_clients(
          i, num_clients=self.clients_per_round)
      np.random.seed(i)
      cpr = self.clients_per_round
      if cpr > len(selected_clients):
        cpr = len(selected_clients)
      active_clients = np.random.choice(selected_clients, round(cpr * (1 - self.drop_percent)), replace=False)

      # buffer for receiving client solutions
      cws = []

      # communicate the latest model
      inner_loop = tqdm(
          total=len(active_clients), desc='Client', position=1,
          dynamic_ncols=True)
      self.distribute(active_clients)
      # Logging.
      # rewards_before = self.test(active_clients)[2]
      # An experiment about the performance of FedTRPO if \rho are not confidential.
      # Remember to call distribute() before this step.
      _, norm_penalties = self.get_state_visitation_frequency(
          active_clients, logger=outer_loop.write if self.verbose \
              else None)
      # Commence this round.
      for idx, c in enumerate(active_clients):
        # Sync local (global) params to local old policy before training.
        # c.sync_old_policy()
        # Enable svf so as to calculate norm penalty.
        c.enable_svf(self.svf_n_timestep)
        # Sequentially train each client. Notice that, we do not sync old
        # policy before each local fit, but before round.
        self.retry(
            [
                lambda: self.distribute([c]),
                lambda: c.reset_client_weight(),
                # sync local (global) params to local anchor.
                lambda: c.sync_anchor_policy(),
            ],
            lambda: c.experiment(
                num_iter=self.num_iter,
                timestep_per_batch=self.timestep_per_batch,
                callback_before_fit=[c.sync_old_policy],
                logger=inner_loop.write if verbose else None,
                norm_penalty=norm_penalties[idx:idx + 1]
            ),
            logger=inner_loop.write if verbose else None,
        )
        # gather weights from client
        cws.append((c.get_client_weight(), c.get_params()))

        # track communication cost
        # self.metrics.update(rnd=i, cid=c.id, stats=stats)
        inner_loop.update()

      # update models
      self.global_weights = self.aggregate(cws)

      # rewards_after = self.test(active_clients)[2]
      # if i % self.eval_every == 0:
      #   inner_loop.write(
      #       'At round {}, group expected future discounted reward before training: {}, after training: {}'.format(
      #           i, np.mean(rewards_before), np.mean(rewards_after)))

      outer_loop.update()

    # final test model
    stats = self.test()
    rewards = stats[2]
    reward_history.append(rewards)
    logging.error('At round {} total reward received: {}'.format(self.num_rounds, np.mean(rewards)))
    return reward_history
