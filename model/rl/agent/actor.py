#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from absl import logging
from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import model.rl.agent.ops as ops_lib
import model.rl.comp.cfn_cell as cfn_lib
import model.utils.utils as utils_lib

tfv1.disable_v2_behavior()
# Need to enable TFv2 control flow with support for higher order derivatives
# in keras LSTM layer.
tfv1.enable_control_flow_v2()


class Actor(ABC):
  @abstractmethod
  def build_network(self, observations, actions, next_observations, seq_mask):
    pass

  @abstractmethod
  def build_loss(self, reward, actions, logits_pi,
                 prob_pi, logits_beta, prob_beta):
    pass

  @abstractmethod
  def build_opti_ops(self, pi_var, loss_pi, beta_var,
                     loss_beta, grad_tape):
    pass
