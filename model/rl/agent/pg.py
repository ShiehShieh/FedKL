#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from absl import logging

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

tfv1.disable_v2_behavior()
# Need to enable TFv2 control flow with support for higher order derivatives
# in keras LSTM layer.
tfv1.enable_control_flow_v2()


class PolicyGradient(object):
  def __init__(self):
    self.num_fit = 0.0
    self.num_timestep_seen = 0.0

  def build_network(self, observations, actions, next_observations, seq_mask):
    raise NotImplementedError

  def build_loss(self, reward, actions, logits_pi,
                 prob_pi, logits_beta, prob_beta):
    raise NotImplementedError

  def build_opti_ops(self, pi_var, loss_pi, beta_var,
                     loss_beta, grad_tape):
    raise NotImplementedError

  def stat(self):
    raise NotImplementedError

  def set_params(self, model_params=None):
    if model_params is None:
      return
    with self.graph.as_default():
      all_vars = tfv1.trainable_variables()
      for variable, value in zip(all_vars, model_params):
        variable.load(value, self.sess)

  def get_params(self, var_list=None):
    with self.graph.as_default():
      if var_list is not None:
        return self.sess.run(var_list)
      tvars = tfv1.trainable_variables()
      model_params = self.sess.run(tvars)
    return model_params

  def reset_num_timestep_seen(self):
    self.num_timestep_seen = 0.0

  def get_num_timestep_seen(self):
    return self.num_timestep_seen
