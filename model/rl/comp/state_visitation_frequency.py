#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
from collections import defaultdict, OrderedDict

import tensorflow as tf
import tensorflow.compat.v1 as tfv1


def round_resolution(x, resolution):
  return np.round(x / resolution) * resolution


def find_discounted_svf(n_states, trajectories, svf_m=None, gamma=1.0):
  # Continuous state space.
  # OrderedDict(sorted(d.items()))
  if n_states == -1:
    seq_len = [t['observations'].shape[0] for t in trajectories]
    max_seq_len = np.max(seq_len)
    mask = np.array([[float(j < sl) for j in range(max_seq_len)]
                     for i, sl in enumerate(seq_len)])
    # pr = mask / mask.sum(axis=0)
    pr = mask / mask.sum()

    d = defaultdict(float)
    a = defaultdict(lambda: defaultdict(float))
    acnt = defaultdict(lambda: defaultdict(float))
    summation = 0.0
    for i, trajectory in enumerate(trajectories):
      for j, obs in enumerate(trajectory['observations']):
        act = trajectory['actions'][j]
        # act = np.round(act, decimals=1)
        # obs = np.round(obs, decimals=1)
        act = np.round(act, decimals=1)
        obs = np.round(obs, decimals=0)
        # act = round_resolution(act, resolution=0.1)
        # obs = round_resolution(obs, resolution=0.01)
        # obsk = ','.join(map(str, obs.tolist()))
        # actk = ','.join(map(str, act.tolist()))
        obsk = tuple(obs.tolist())
        actk = tuple(act.tolist())
        d[obsk] += pow(gamma, j) * pr[i, j]
        # a[obsk][actk] += trajectory['prob_advs'][j]
        a[obsk][actk] += trajectory['advantages'][j]
        acnt[obsk][actk] += 1
        summation += d[obsk]
    for k, v in d.items():
      # d[k] = (1 - gamma) * v
      d[k] = (1 / summation) * v
    for k, kv in a.items():
      for kk, vv in kv.items():
        a[k][kk] /= acnt[k][kk]

    return d, a


def find_svf(n_states, trajectories, svf_m=None):
    """
    Find the state vistiation frequency from trajectories.
    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    """
    # Continuous state space.
    if n_states == -1:
      m = defaultdict(float)
      if svf_m is not None:
        m = svf_m

      for trajectory in trajectories:
        for obs in trajectory['observations']:
          obs = np.round(obs, decimals=0)
          m[tuple(obs.tolist())] += 1.0

      return np.array([v for k, v in m.items()]) / float(len(trajectories)), m

    # Finite state space.
    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for obs in trajectory['observations']:
            svf[obs] += 1

    svf /= trajectories.shape[0]

    return svf


def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.
    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    # policy = find_policy(n_states, r, n_actions, discount,
    #                                 transition_probability)
    policy = value_iteration.find_policy(n_states, n_actions,
                                         transition_probability, r, discount)

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory[0, 0]] += 1
    p_start_state = start_state_count/n_trajectories

    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] * # Stochastic policy
                                  transition_probability[i, j, k])

    return expected_svf.sum(axis=1)
