#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.signal

import tensorflow as tf
import tensorflow.compat.v1 as tfv1


def remove_first_step(data, axis=1):
  shape = tf.shape(data)
  ndims = data.get_shape().ndims
  begin = [0] * ndims
  begin[axis] = 1
  size = [shape[i] for i in range(ndims)]
  size[axis] -= 1
  # Assuming (batch_size, seq_len, ...)
  shape_list = data.shape.as_list()
  out = tf.slice(data, begin, size)
  if shape_list[-1] is not None:
    nones = [None] * (len(shape_list) - 1)
    out.set_shape(nones + [shape_list[-1]])
  return out


def remove_last_step(data, axis=1):
  shape = tf.shape(data)
  ndims = data.get_shape().ndims
  begin = [0] * ndims
  size = [shape[i] for i in range(ndims)]
  size[axis] -= 1
  # Assuming (batch_size, seq_len, ...)
  shape_list = data.shape.as_list()
  out = tf.slice(data, begin, size)
  if shape_list[-1] is not None:
    nones = [None] * (len(shape_list) - 1)
    out.set_shape(nones + [shape_list[-1]])
  return out


# TODO(jay.xie): remove_first_unmasked_step?
def remove_last_unmasked_step(data, seq_mask, axis=1):
  """
  In case seq_mask is all False, the second step will be removed, because
  argidx_last_unmasked returns 0th.
  """
  shape = tf.shape(data)
  ndims = data.get_shape().ndims
  # set the step behind the last unmasked step to zero.
  last_true_step_add1 = tf.squeeze(
      argidx_last_unmasked(seq_mask), axis=1) + 1
  oh = tf.one_hot(
      last_true_step_add1, depth=shape[1], dtype=data.dtype)
  if ndims == 3:
    oh = tf.tile(
        tf.expand_dims(oh, axis=2), multiples=[1, 1, shape[2]])
  data = data - data * oh
  return remove_last_step(data, axis)


def append_last_unmasked_step(data, seq_mask, value=0, constant_values=0, axis=1):
  data = tf.pad(
      data, [[0, 0], [0, 1]], "CONSTANT",
      constant_values=constant_values)
  shape = tf.shape(data)
  ndims = data.get_shape().ndims
  # set the step behind the last unmasked step to value.
  last_true_step_add1 = tf.squeeze(
      argidx_last_unmasked(seq_mask), axis=1) + 1
  oh = tf.one_hot(
      last_true_step_add1, depth=shape[1], dtype=data.dtype)
  if ndims == 3:
    oh = tf.tile(
        tf.expand_dims(oh, axis=2), multiples=[1, 1, shape[2]])
  return data + value * oh


def argidx_last_unmasked(seq_mask):
  seq_mask_range = tf.tile(
      tf.expand_dims(
          tf.range(0, tf.shape(seq_mask)[1], dtype=tf.int32),
          axis=0,
      ),
      multiples=[tf.shape(seq_mask)[0], 1],
  )
  return tf.reduce_max(
      tf.cast(seq_mask, tf.int32) * seq_mask_range,
      axis=1, keepdims=True,
  )


def __num_elems(shape):
  '''Returns the number of elements in the given shape
  Args:
      shape: TensorShape
  
  Return:
      tot_elems: int
  '''
  tot_elems = 1
  for s in shape:
    tot_elems *= int(s)
  return tot_elems


def graph_size(graph):
  '''Returns the size of the given graph in bytes
  The size of the graph is calculated by summing up the sizes of each
  trainable variable. The sizes of variables are calculated by multiplying
  the number of bytes in their dtype with their number of elements, captured
  in their shape attribute
  args:
      graph: tf graph
  return:
      integer representing size of graph (in bytes)
  '''
  tot_size = 0
  with graph.as_default():
    vs = tfv1.trainable_variables()
    for v in vs:
      tot_elems = __num_elems(v.shape)
      dtype_size = int(v.dtype.size)
      var_size = tot_elems * dtype_size
      tot_size += var_size
  return tot_size


def discount(x, gamma):
  """ Calculate discounted forward sum of a sequence at each point """
  return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def generalized_advantage_estimate(vf, trajectories, gamma, lam):
  """ Add generalized advantage estimator.
  https://arxiv.org/pdf/1506.02438.pdf
  Args:
      trajectories: as returned by run_policy(), must include 'values'
          key from add_value().
      gamma: reward discount
      lam: lambda (see paper).
          lam=0 : use TD residuals
          lam=1 : A =  Sum Discounted Rewards - V_hat(s)
  Returns:
      None (mutates trajectories dictionary to add 'advantages')
  """
  for trajectory in trajectories:
    for k, v in trajectory.items():
      trajectory[k] = np.array(v)
    trajectory['value'] = vf(trajectory['observations'])
    rewards = trajectory['reward']
    values = trajectory['value']
    # temporal differences
    tds = rewards - values + np.append(values[1:] * gamma, 0)
    advantages = discount(tds, gamma * lam)
    trajectory['advantages'] = advantages
  alladv = np.concatenate(
      [trajectory["advantages"] for trajectory in trajectories])
  # Standardize advantage
  std = alladv.std()
  mean = alladv.mean()
  for trajectory in trajectories:
    trajectory['advantages'] = (trajectory['advantages'] - mean) / std


def standardize(x):
  std = x.std()
  mean = x.mean()
  return (x - mean) / std


def convert_trajectories_to_steps(trajectories, shuffle=False):
  o = {
      'observations': np.concatenate(
          [t['observations'] for t in trajectories], axis=0),
      'actions': np.concatenate(
          [t['actions'] for t in trajectories], axis=0),
      'reward': np.concatenate(
          [t['reward'] for t in trajectories], axis=0),
      'dfr': np.concatenate(
          [t['dfr'] for t in trajectories], axis=0),
      'advantages': np.concatenate(
          [t['advantages'] for t in trajectories], axis=0),
      'seq_mask': np.concatenate(
          [t['seq_mask'] for t in trajectories], axis=0),
      'probs': np.concatenate(
          [t['probs'] for t in trajectories], axis=0),
      'value': np.concatenate(
          [t['value'] for t in trajectories], axis=0),
  }
  if shuffle:
    return shuffle_map(o)
  return o

def shuffle_map(m):
  p = np.random.permutation(len(m[list(m.keys())[0]]))
  for k, v in m.items():
    if v.shape[0] != p.shape[0]:
      continue
    m[k] = v[p]
  return m

def stablize(x):
  eps = 1e-8
  return tf.cast(tf.math.less_equal(x, eps), tf.float32) * eps + x

def categorical_kl(prob0, prob1):
  return tf.reduce_sum(
      (prob0 * tf.math.log(
          stablize(prob0) / stablize(prob1)
      )
    ), axis=1
  )

def multivariate_normal_kl(prob0, prob1):
  mean0 = prob0[:, :self.d]
  std0 = prob0[:, self.d:]
  mean1 = prob1[:, :self.d]
  std1 = prob1[:, self.d:]
  return tf.reduce_sum(tf.math.log(std1 / std0), axis=1) + \
      tf.reduce_sum(((tf.math.square(std0) + tf.math.square(mean0 - mean1)) / (2.0 * tf.math.square(std1))), axis=1) - \
      0.5 * self.d
