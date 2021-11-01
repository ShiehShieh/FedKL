#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import model.rl.agent.pg as pg_lib
import model.utils.utils as utils_lib
import model.utils.distributions as distributions_lib

tfv1.disable_eager_execution()

tfv1.disable_v2_behavior()
# Need to enable TFv2 control flow with support for higher order derivatives
# in keras LSTM layer.
tfv1.enable_control_flow_v2()


class PolicyNN(object):
  def __init__(self, num_actions, dropout_rate=None):
    self.dropout_rate = dropout_rate
    self.l = tf.keras.layers.Dense(
        units=128, use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        kernel_initializer='glorot_normal',
        name="policy_network_l")
    self.o = tf.keras.layers.Dense(
        units=num_actions, use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        kernel_initializer='glorot_normal',
        name="policy_network_o")

  def forward(self, observations, actions):
    l = tf.keras.layers.Activation('tanh')(
        self.l(observations)
    )
    # if self.dropout_rate is not None:
    #   l = tf.keras.layers.Dropout(rate=self.dropout_rate)(l)
    logits = self.o(l)
    return logits


class REINFORCEActor(pg_lib.PolicyGradient):
  def __init__(self,
               env,
               optimizer,
               model_scope,
               batch_size=16,
               future_discount=0.99,
               normalize_reward=False,
               importance_weight_cap=10.0,
               dropout_rate=0.0):
    super(REINFORCEActor, self).__init__()

    self.optimizer = optimizer
    # training parameters
    self.state_dim             = env.state_dim
    self.num_actions           = env.num_actions
    self.future_discount       = future_discount
    self.batch_size            = batch_size
    self.dropout_rate          = dropout_rate
    self.normalize_reward      = normalize_reward
    self.importance_weight_cap = importance_weight_cap
    self.env_sample            = env.env_sample
    self.output_types          = env.output_types
    self.output_shapes         = env.output_shapes

    self.graph = tf.Graph()
    with self.graph.as_default():
      with tfv1.variable_scope(model_scope, default_name='reinforce') as vs, tf.GradientTape(persistent=True) as gt:

        self.observations = tfv1.placeholder(
            self.output_types['observations'],
            self.output_shapes['observations'], name='observations')
        self.actions = tfv1.placeholder(
            self.output_types['actions'],
            self.output_shapes['actions'], name='actions')
        self.seq_mask = tfv1.placeholder(
            self.output_types['seq_mask'],
            self.output_shapes['seq_mask'], name='seq_mask')
        self.reward = tfv1.placeholder(
            self.output_types['reward'],
            self.output_shapes['reward'], name='reward')
        self.dfr = tfv1.placeholder(
            self.output_types['dfr'],
            self.output_shapes['dfr'], name='dfr')

        self.policy_network = PolicyNN(self.num_actions)
        logits_pi, prob_pi, pi_var = self.build_network(
            self.observations, self.actions, self.seq_mask)
        prob_pi_t = self.build_loss(
            self.reward, self.actions, logits_pi, prob_pi, self.seq_mask)
        train_op, pi_grad = self.build_opti_ops(prob_pi_t, pi_var, gt)

        # Keep handy fields.
        self.train_op = train_op
        self.gradients = pi_grad
        self.prob_pi = prob_pi

    self.sess = tfv1.Session(graph=self.graph, config=tfv1.ConfigProto(log_device_placement=True))

    # find memory footprint and compute cost of the model
    self.graph_size = utils_lib.graph_size(self.graph)
    with self.graph.as_default():
      self.sess.run(tfv1.global_variables_initializer())
      self.sess.run(tfv1.tables_initializer())
      # metadata = tfv1.RunMetadata()
      # opts = tfv1.profiler.ProfileOptionBuilder.float_operation()
      # self.flops = tfv1.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

  def build_network(self, observations, actions, seq_mask):
    # Define trainable variables.
    self.global_step = tf.Variable(0, name="global_step")
    logits_pi = self.policy_network.forward(observations, actions)

    pi_var = self.policy_network.l.trainable_variables + self.policy_network.o.trainable_variables

    prob_pi = tf.nn.softmax(logits_pi)

    return logits_pi, prob_pi, pi_var

  def build_loss(self, reward, actions, logits_pi, prob_pi, seq_mask):
    # Epsilon for numerical stability.
    eps = 1e-8
    # Get prediction on t.
    prob_pi_t = tf.gather(prob_pi, actions, batch_dims=-1)

    # Calculate Pi loss.
    obj_pi = tf.stop_gradient(self.dfr) * tf.math.log(tf.maximum(prob_pi_t, eps))

    return -obj_pi

  def build_opti_ops(self, prob_pi_t, pi_var, grad_tape):
    # compute gradients
    pi_grad = tf.gradients(prob_pi_t, pi_var)
    pi_grad = [(pi_grad[i], pi_var[i]) for i in range(len(pi_var))]
    pi_op = self.optimizer.apply_gradients(pi_grad)

    new_global_step = self.global_step + 1
    train_op = tf.group([pi_op, self.global_step.assign(new_global_step)])

    return train_op, pi_grad

  def fit(self, steps, logger=None):
    self.num_timestep_seen += float(len(steps['observations']))
    for i in range(len(steps['observations']))[::self.batch_size]:
      end = i + self.batch_size
      m = {
          self.observations: [step for step in steps['observations'][i:end]],
          self.actions: [step for step in steps['actions'][i:end]],
          self.reward: [[step] for step in steps['reward'][i:end]],
          self.dfr: [step for step in steps['dfr'][i:end]],
          self.seq_mask: [1 for step in steps['dfr'][i:end]],
      }

      # perform one update of training
      _, global_step = self.sess.run([
          self.train_op,
          self.global_step,
        ], feed_dict=m
      )

  def act(self, observations, stochastic=True):
    m = {
        self.observations: observations,
        self.actions: self.env_sample['actions'],
        self.reward: self.env_sample['reward'],
        self.seq_mask: self.env_sample['seq_mask'],
        self.dfr: self.env_sample['dfr'],
    }
    probs = self.sess.run(self.prob_pi, feed_dict=m)
    if stochastic:
      return distributions_lib.categorical_sample(probs), probs
    return np.argmax(probs, axis=1), probs
