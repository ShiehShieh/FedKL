from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
from tqdm import tqdm

import sys
import random
import numpy as np

import model.fl.fedbase as fedbase_lib
import model.rl.agent.vec_agent as vec_agent_lib
import model.utils.vectorization as vectorization_lib


class FedTRPO(fedbase_lib.FederatedBase):

  def __init__(self, clients_per_round, num_rounds, num_iter,
               timestep_per_batch, max_steps, eval_every, drop_percent,
               verbose=False, svf_n_timestep=1e6, has_global_svf=False,
               kl_targ_adap=(0.5, 0.3, 20.0),
               retry_min=-sys.float_info.max, universial_client=None,
               reward_history_fn='', **kwargs):
    super(FedTRPO, self).__init__(
        clients_per_round, num_rounds, num_iter, timestep_per_batch,
        max_steps, eval_every, drop_percent, retry_min, universial_client,
        reward_history_fn)
    self.verbose = verbose
    self.svf_n_timestep = svf_n_timestep
    self.has_global_svf = has_global_svf
    self.kl_targ_adap = kl_targ_adap

  def get_state_visitation_frequency(self, active_clients, logger=None):
    return None, [None] * len(active_clients)

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

    if self.has_global_svf:
      avg = np.mean(svfs, axis=0)
      norm_penalties = np.linalg.norm(avg - svfs, ord=2, axis=1)
    else:
      norm_penalties = np.linalg.norm(svfs, ord=2, axis=1)

    if logger:
      logger('norm_penalties shape %s, l2 norm: %s' % (
          norm_penalties.shape, norm_penalties))
    return svfs, norm_penalties

  def _inner_sequential_loop(self, i_iter, active_clients, retry_min):
    verbose = self.verbose
    # buffer for receiving client solutions
    cws = []
    inner_loop = tqdm(
        total=len(active_clients), desc='Client', position=1,
        dynamic_ncols=True)
    logger = lambda x: inner_loop.write(x, file=sys.stderr)
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
              logger=logger if verbose else None,
              # norm_penalty=norm_penalties[idx:idx + 1],
          ),
          max_retry=5 if i_iter > 3 else 0,
          logger=logger if verbose else None,
          retry_min=retry_min - np.abs(retry_min),
      )
      # gather weights from client
      cws.append((c.get_client_weight(), c.get_params()))

      # track communication cost
      # self.metrics.update(rnd=i, cid=c.cid, stats=stats)
      inner_loop.update()
    return cws

  def _inner_vectorized_loop(self, i_iter, indices, retry_min):
    verbose = self.verbose
    # Create vectorized objects.
    active_clients = [self.clients[idx] for idx in indices]
    # buffer for receiving client solutions
    cws = []
    # Commence this round.
    for c in active_clients:
      self.distribute([c])
      c.reset_client_weight()
      c.sync_anchor_policy()
    self.universial_client.experiment(
        num_iter=self.num_iter,
        timestep_per_batch=self.timestep_per_batch, indices=indices,
        agents=self.agents, obfilts=self.obfilts, rewfilts=self.rewfilts,
        callback_before_fit=[c.sync_old_policy for c in active_clients],
        logger=print if verbose else None,
    )
    # gather weights from client
    for c in active_clients:
      cws.append((c.get_client_weight(), c.get_params()))
    return cws
