#!/usr/bin/env python

from absl import logging

import numpy as np
from collections import defaultdict

import model.utils.utils as utils_lib


class VecCallable(object):
  def __init__(self, funcs):
    self.funcs = funcs
    self.num_funcs = len(funcs)

  def __call__(self, xs):
    assert len(xs) == self.num_funcs or self.num_funcs == 1, 'mismatched xs dim and # funcs: want: %d, got: %d' % (self.num_funcs, len(xs))
    if self.num_funcs == 1:
      return [self.funcs[0](x) for x in xs]
    out = []
    for i in range(self.num_funcs):
      x = xs[i]
      func = self.funcs[i]
      out.append(func(x))
    return out

  def call(self, i, x):
    return self.funcs[i](x)


def path_length(trajectory):
  return len(trajectory['actions'])


def vectorized_rollout(agents, envs, obfilts, rewfilts, future_discount, lam, n_timesteps=-1, n_episodes=-1, is_per=False, extra_features=set([]), logger=None):
  if (n_timesteps < 0 and n_episodes < 0) or \
      (n_timesteps > 0 and n_episodes > 0):
    raise Exception('either n_timesteps: [%s] or n_episodes: [%s] should be larger than 0' % (n_timesteps, n_episodes))

  paths = []
  paths_list = [[] for _ in range(envs.num_envs)]
  timesteps_sofar = 0
  timesteps_per = np.zeros(shape=(envs.num_envs,), dtype=int).tolist()
  episode_rewards = [[] for _ in range(envs.num_envs)]
  while True:
    states = envs.reset()
    states = np.array(obfilts(states))
    total_rewards = np.zeros(shape=(envs.num_envs,)).tolist()

    # Generate trajectory.
    trajectories = [defaultdict(list) for i in range(envs.num_envs)]
    already_done = np.zeros(shape=(envs.num_envs,), dtype=bool)
    is_corrupt = False
    while True:
      actions, probses = agents.act(states)
      next_states, rewards, dones, infos = envs.step(actions)
      if 'err' in infos[0]:
        is_corrupt = True
        if logger:
          logger("infos[0]: {}".format(infos[0]))
        break

      # Process each env seperately.
      for i, trajectory in enumerate(trajectories):
        next_states[i] = obfilts.call(i, next_states[i])
        if already_done[i]:
          # This env has finished its episode here.
          continue
        already_done[i] = dones[i]
        total_rewards[i] += rewards[i]
        rewards[i] = rewfilts.call(i, rewards[i])
        trajectory['observations'].append(states[i])
        trajectory['actions'].append(actions[i])
        trajectory['reward'].append(rewards[i])
        if 'next_observations' in extra_features:
          trajectory['next_observations'].append(next_states[i])
        if 'probs' in extra_features:
          trajectory['probs'].append(probses[i])

      states = next_states
      if np.all(dones):
        break
      if is_per and n_timesteps > 0 and \
          np.min([timesteps_per[i] + path_length(t)
                  for i, t in enumerate(trajectories)]) > n_timesteps:
        break

    if is_corrupt:
      continue

    for i, tr in enumerate(total_rewards):
      episode_rewards[i].append(tr)
    for i, trajectory in enumerate(trajectories):
      trajectory['dfr'] = utils_lib.discount(
          trajectory['reward'], future_discount)
      paths.append(trajectory)
      paths_list[i].append(trajectory)
      timesteps_per[i] += path_length(trajectory)
      timesteps_sofar += path_length(trajectory)
    if not is_per and n_timesteps > 0 and timesteps_sofar >= n_timesteps:
      break
    if not is_per and n_episodes > 0 and len(paths) >= n_episodes:
      paths = paths[:n_episodes]
      break
    if is_per and n_timesteps > 0 and np.min(timesteps_per) >= n_timesteps:
      break
    if is_per and n_episodes > 0 and np.min(
        [len(p) for p in paths_list]) >= n_episodes:
      for i in range(len(paths_list)):
        paths_list[i] = paths_list[i][:n_episodes]
        episode_rewards[i] = episode_rewards[i][:n_episodes]
      paths = [item for sub in paths_list for item in sub]
      break

  for i, ps in enumerate(paths_list):
    utils_lib.generalized_advantage_estimate(
        agents.get_value_func(i), ps, future_discount, lam)
  return paths, paths_list, episode_rewards
