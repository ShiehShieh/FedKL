from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
from tqdm import tqdm

import sys
import csv
import random
import numpy as np

from collections import defaultdict

from mujoco_py.builder import MujocoException

import model.rl.agent.critic as critic_lib
import model.rl.agent.vec_agent as vec_agent_lib
import model.utils.vectorization as vectorization_lib


class FederatedBase(object):

  def __init__(self, clients_per_round, num_rounds, num_iter,
               timestep_per_batch, max_steps, eval_every, drop_percent,
               retry_min=-sys.float_info.max, universial_client=None,
               eval_heterogeneity=False, reward_history_fn='',
               b_history_fn='', da_history_fn='', avg_history_fn=''):
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
    self.b_history_fn = b_history_fn
    self.da_history_fn = da_history_fn
    self.avg_history_fn = avg_history_fn
    self.universial_client = universial_client
    self.eval_heterogeneity = eval_heterogeneity
    self.afs = []

  def register_universal_client(self, universial_client):
    self.universial_client = universial_client

  def register(self, client):
    self.clients.append(client)
    if self.global_weights is None:
      self.global_weights = client.get_params()
    # Create vectorized objects.
    self.agents = vec_agent_lib.VecAgent([c.agent for c in self.clients])
    self.obfilts = vectorization_lib.VecCallable(
        [c.obfilt for c in self.clients])
    self.rewfilts = vectorization_lib.VecCallable(
        [c.rewfilt for c in self.clients])
    n = 13
    self.afs.append(critic_lib.Critic(n, 200, seed=0, lr=1e-2, epochs=30))
    # from sklearn.linear_model import ElasticNet
    # from sklearn.neural_network import MLPRegressor
    # self.afs.append(ElasticNet(alpha=0.1, l1_ratio=0.1))
    # self.afs.append(MLPRegressor(
    #     hidden_layer_sizes=(100, 100),learning_rate_init=1e-3,
    # ))

  def num_clients(self):
    return len(self.clients)

  def get_client(self, i):
    return self.clients[i]

  def distribute(self, clients):
    for client in clients:
      client.set_params(self.global_weights)

  def select_clients(self, round_id, num_clients):
    num_clients = min(num_clients, len(self.clients))
    # make sure for each comparison, we are selecting the same clients each round
    # np.random.seed(round_id)
    np.random.seed(round_id + 1000)  #
    # np.random.seed(round_id + 10000)  #
    indices = np.random.choice(
        range(len(self.clients)), num_clients, replace=False)
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

  def _inner_vectorized_loop(self, i_iter, indices, retry_min):
    raise NotImplementedError

  def train(self):
    logging.error('Training with {} workers per round ---'.format(
        self.clients_per_round))
    retry_min = self.retry_min
    reward_history = []
    b_history = []
    da_history = []
    avg_history = []
    outer_loop = tqdm(
        total=self.num_rounds, desc='Round', position=0,
        dynamic_ncols=True)
    logger = lambda x: outer_loop.write(x, file=sys.stderr)
    for i in range(self.num_rounds):
      # test model
      if i % self.eval_every == 0:
        if self.universial_client is not None:
          stats = self.universal_test()
        else:
          stats = self.test(logger=logger)  # have distributed the latest model.
        rewards = stats[2]
        norm_bs = stats[3]
        norm_das = stats[4]
        norm_avgs = stats[5]
        retry_min = np.mean(rewards)
        reward_history.append(rewards)
        b_history.append(norm_bs)
        da_history.append(norm_das)
        avg_history.append(norm_avgs)
        self.log_csv(reward_history, self.reward_history_fn)
        if self.eval_heterogeneity:
          if len(self.b_history_fn) > 0:
            self.log_csv(b_history, self.b_history_fn)
          if len(self.da_history_fn) > 0:
            self.log_csv(da_history, self.da_history_fn)
          if len(self.avg_history_fn) > 0:
            self.log_csv(avg_history, self.avg_history_fn)
        outer_loop.write(
            'At round {} expected future discounted reward: {}, averaged level of heterogeneity B norm: {}; averaged [D]xA norm: {}; norm of \mean DxA: {}; # retry so far {}'.format(
                i, np.mean(rewards), np.mean(norm_bs), np.mean(norm_das),
                np.mean(norm_avgs), self.get_num_retry()),
            file=sys.stderr)

      # uniform sampling
      indices, selected_clients = self.select_clients(
          i, num_clients=self.clients_per_round)
      np.random.seed(i)
      cpr = self.clients_per_round
      if cpr > len(selected_clients):
        cpr = len(selected_clients)
      active_clients = np.random.choice(
          selected_clients, round(cpr * (1 - self.drop_percent)),
          replace=False)

      # communicate the latest model
      self.distribute(active_clients)
      # buffer for receiving client solutions
      cws = []
      # Inner sequantial loop.
      if self.universial_client is not None:
        cws = self._inner_vectorized_loop(i, indices, retry_min)
      else:
        cws = self._inner_sequential_loop(i, active_clients, retry_min)

      # update models
      self.global_weights = self.aggregate(cws)

      outer_loop.update()

    # final test model
    if self.universial_client is not None:
      stats = self.universal_test()
    else:
      stats = self.test(logger=logger)  # have distributed the latest model.
    rewards = stats[2]
    reward_history.append(rewards)
    self.log_csv(reward_history, self.reward_history_fn)
    outer_loop.write(
        'At round {} total reward received: {}'.format(
            self.num_rounds, np.mean(rewards)),
        file=sys.stderr)
    return reward_history

  def test(self, clients=None, logger=None):
    self.distribute(self.clients)
    rewards = []
    if clients is None:
      clients = self.clients
    d_as = []
    for c in clients:
      r = self.retry(
          [],
          lambda: c.test(),
          max_retry=5,
          logger=None,
          retry_min=-sys.float_info.max,
      )
      rewards.append(r)
      if self.eval_heterogeneity:
        d, a = c.get_da()
        d_as.append((d, a))
    ids = [c.cid for c in self.clients]
    groups = [c.group for c in self.clients]
    norm_bs = [0.0] * len(self.clients)
    norm_das = [0.0] * len(self.clients)
    norm_avgs = [0.0] * len(self.clients)
    if self.eval_heterogeneity:
      norm_bs, norm_das, norm_avgs = self.get_heterogeneity_level(
          d_as, logger)
    return ids, groups, rewards, norm_bs, norm_das, norm_avgs

  def universal_test(self):
    self.distribute(self.clients)
    rewards = self.universial_client.test(self.agents, self.obfilts,
                                          self.rewfilts)
    ids = [c.cid for c in self.clients]
    groups = [c.group for c in self.clients]
    norm_bs = [0.0] * len(self.clients)
    norm_das = [0.0] * len(self.clients)
    norm_avgs = [0.0] * len(self.clients)
    return ids, groups, rewards, norm_bs, norm_das, norm_avgs

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

  def get_heterogeneity_level(self, d_as, logger=None):
    all_keys = set()
    for i, d_a in enumerate(d_as):
      d, a = d_a
      xs, ys = [], []
      for obsk, kv in a.items():
        for actk, adv in kv.items():
          k = obsk + actk
          all_keys.add(k)
          xs.append(k)
          ys.append(adv)
      xs = np.array(xs)
      ys = np.array(ys)
      self.afs[i].fit(xs, ys)
      loss = 0.0
      for j, x in enumerate(xs):
        p = self.afs[i].predict([x])
        loss += np.power(p - ys[j], 2)
      loss /= float(len(xs))
      if logger:
        logger('%d: # samples: %d, advantage function loss: %.10f' % (i, len(xs), loss))
        ys = np.abs(ys)
        logger('%f, %f, %f' % (np.mean(ys), np.min(ys), np.max(ys)))
    for i, d_a in enumerate(d_as):
      a = d_a[1]
      local_keys = set([k + kk for k in a for kk in a[k]])
      missed = tuple(all_keys - local_keys)
      for idx in range(len(missed) // 128):
        start = idx * 128
        end = start + 128
        if idx == (len(missed) // 128) - 1:
          end = len(missed)
        vs = self.afs[i].predict(missed[start:end])
        for j, v in enumerate(vs):
          if np.abs(v) < 0:
            # continue
            pass
          k = missed[start:end][j]
          obsk, actk = k[:-2], k[-2:]
          a[obsk][actk] = v
      # local_keys = set([k + kk for k in a for kk in a[k]])

    # Merge obsk and actk.
    for i, d_a in enumerate(d_as):
      d, a = d_a
      out = defaultdict(float)
      for obsk in a:
        mu = d[obsk]
        for actk in a[obsk]:
          adv = a[obsk][actk]
          if obsk in out and actk in out[obsk]:
            exit(0)
          out[obsk + actk] = mu * adv
      d_as[i] = out
    das = d_as

    # joint_keys = set(das[0].keys())
    # for da in das[1:]:
    #   for k in tuple(joint_keys):
    #     if k not in da:
    #       joint_keys.remove(k)
    # if logger:
    #   logger('# joint keys: %d.' % (len(joint_keys), ))
    # for da in das:
    #   for k in tuple(da.keys()):
    #     if k not in joint_keys:
    #       da.pop(k)

    avg = defaultdict(float)
    for da in das:
      for k in da:
        v = da[k]
        avg[k] += v / float(len(das))
    norm_avg = 0.0
    for k in avg:
      g = avg[k]
      norm_avg += pow(g, 2)
    norm_avgs = [np.sqrt(norm_avg)] * len(das)

    norm_bs = []
    norm_das = []
    for da in das:
      norm_b = 0.0
      norm_da = 0.0
      for k in da:
        # for idx, da in enumerate(das):
        g = avg[k]
        v = da[k]
        norm_b += pow(g - v, 2)
        norm_da += pow(v, 2)
      norm_bs.append(np.sqrt(norm_b))
      norm_das.append(np.sqrt(norm_da))

    return norm_bs, norm_das, norm_avgs

  # def get_heterogeneity_level(self, d_as, logger=None):
  #   all_keys = set()
  #   for i, d_a in enumerate(d_as):
  #     d, a = d_a
  #     xs, ys = [], []
  #     for obsk, kv in a.items():
  #       for actk, adv in kv.items():
  #         k = obsk + actk
  #         all_keys.add(k)
  #         xs.append(k)
  #         ys.append(adv)
  #     xs = np.array(xs)
  #     ys = np.array(ys)
  #     self.afs[i].fit(xs, ys)
  #     loss = 0.0
  #     for j, x in enumerate(xs):
  #       p = self.afs[i].predict([x])
  #       loss += np.power(p - ys[j], 2)
  #     loss /= float(len(xs))
  #     if logger:
  #       logger('%d: # samples: %d, advantage function loss: %.10f' % (i, len(xs), loss))
  #       ys = np.abs(ys)
  #       logger('%f, %f, %f' % (np.mean(ys), np.min(ys), np.max(ys)))
  #   for i, d_a in enumerate(d_as):
  #     d, a = d_a
  #     local_keys = set([k + kk for k in a for kk in a[k]])
  #     missed = all_keys - local_keys

  #     nulls = set([])
  #     for k in tuple(missed):
  #       obsk = k[:-2]
  #       if d[obsk] == 0.0:
  #         nulls.add(obsk)
  #     missed = missed - nulls

  #     missed = tuple(missed)
  #     for idx in range(len(missed) // 128):
  #       start = idx * 128
  #       end = start + 128
  #       if idx == (len(missed) // 128) - 1:
  #         end = len(missed)
  #       vs = self.afs[i].predict(missed[start:end])
  #       for j, v in enumerate(vs):
  #         if np.abs(v) < 0:
  #           # continue
  #           pass
  #         k = missed[start:end][j]
  #         obsk, actk = k[:-2], k[-2:]
  #         a[obsk][actk] = v

  #   # Merge obsk and actk.
  #   das = [0] * len(d_as)
  #   for i, d_a in enumerate(d_as):
  #     d, a = d_a
  #     out = defaultdict(lambda: defaultdict(float))
  #     for obsk in a:
  #       mu = d[obsk]
  #       for actk in a[obsk]:
  #         adv = a[obsk][actk]
  #         if obsk in out and actk in out[obsk]:
  #           exit(0)
  #         out[obsk][actk] = mu * adv
  #     das[i] = out

  #   avg = defaultdict(lambda: defaultdict(float))
  #   for da in das:
  #     for obsk in da:
  #       for actk in a[obsk]:
  #         v = da[obsk][actk]
  #         avg[obsk][actk] += v / float(len(das))
  #   norm_avg = 0.0
  #   for obsk in avg:
  #     for actk in avg[obsk]:
  #       g = avg[obsk][actk]
  #       norm_avg += pow(g, 2)
  #   norm_avgs = [np.sqrt(norm_avg)] * len(das)

  #   norm_bs = []
  #   norm_das = []
  #   for i, da in enumerate(das):
  #     norm_b = 0.0
  #     norm_da = 0.0
  #     d_a = d_as[i]
  #     d, a = d_a
  #     for obsk in da:
  #       mu = d[obsk]
  #       if mu == 0.0:
  #         continue
  #       for actk in da[obsk]:
  #         g = avg[obsk][actk] / mu
  #         v = da[obsk][actk] / mu
  #         # v = a[obsk][actk]
  #         norm_b += pow(g - v, 2)
  #         norm_da += pow(v, 2)
  #     norm_bs.append(np.sqrt(norm_b))
  #     norm_das.append(np.sqrt(norm_da))

  #   return norm_bs, norm_das, norm_avgs

  def get_num_retry(self):
    return self.num_retry

  def log_csv(self, history, fn):
    if len(fn) == 0:
      raise NotImplementedError('no reward_history_fn and b_history_fn and da_history_fn and avg_history_fn provided')
    with open(fn, 'w', newline='') as csvfile:
      w = csv.writer(csvfile, delimiter=',',
                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
      w.writerows(history)
