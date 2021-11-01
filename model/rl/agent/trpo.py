#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from absl import logging
import numpy as np

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import model.rl.agent.pg as pg_lib
import model.utils.filters as filters_lib
import model.utils.utils as utils_lib
import model.utils.distributions as distributions_lib
import model.utils.prob_type as prob_type_lib

tfv1.disable_eager_execution()

tfv1.disable_v2_behavior()
# Need to enable TFv2 control flow with support for higher order derivatives
# in keras LSTM layer.
tfv1.enable_control_flow_v2()


class DiscretePolicyNN(object):
  def __init__(self, num_actions=1, dropout_rate=None, seed=None):
    self.dropout_rate = dropout_rate
    self.l = tf.keras.layers.Dense(
        units=128, use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
        name="policy_network_l")
    self.o = tf.keras.layers.Dense(
        units=num_actions, use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
        name="policy_network_o")

  def var(self):
    return self.h1.trainable_variables + self.h2.trainable_variables + self.o.trainable_variables
    return self.l.trainable_variables + self.o.trainable_variables

  def forward(self, observations):
    self.h1 = tf.keras.layers.Dense(
        units=64, use_bias=True,
        activation='tanh',
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
        name="h1")
    self.h2 = tf.keras.layers.Dense(
        units=64, use_bias=True,
        activation='tanh',
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
        name="h2")
    logits = self.o(self.h2(self.h1(observations)))
    return tf.nn.softmax(logits)

    l = tf.keras.layers.Activation('tanh')(
        self.l(observations)
    )
    # if self.dropout_rate is not None:
    #   l = tf.keras.layers.Dropout(rate=self.dropout_rate)(l)
    logits = self.o(l)
    return logits


class ContinuousPolicyNN(object):
  def __init__(self, num_actions, dropout_rate=None, seed=None):
    self.dropout_rate = dropout_rate
    self.seed = seed
    self.h1 = tf.keras.layers.Dense(
        units=64, use_bias=True,
        activation='tanh',
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
        name="h1")
    self.h2 = tf.keras.layers.Dense(
        units=64, use_bias=True,
        activation='tanh',
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
        name="h2")
    self.o = tf.keras.layers.Dense(
        units=num_actions, use_bias=True,
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
        name="policy_network_o")
    self.r = tf.Variable(
        shape=(num_actions,), dtype=tf.float32, trainable=True,
        initial_value=[0.0] * num_actions, name='logstd')

  def var(self):
    return self.h1.trainable_variables + self.h2.trainable_variables + self.o.trainable_variables + [self.r]

  def forward(self, observations):
    mean = self.o(self.h2(self.h1(observations)))
    batch_size = tf.shape(observations)[0]
    log_std = tf.tile(tf.expand_dims(self.r, axis=0), (batch_size, 1))
    prob = tf.concat([mean, tf.math.exp(log_std)], axis=1)
    return prob


class TRPOActor(pg_lib.PolicyGradient):
  def __init__(self,
               env,
               optimizer,
               model_scope,
               batch_size=16,
               num_epoch=10,
               future_discount=0.99,
               kl_targ=0.003,
               lam=0.98,
               beta=1.0,
               importance_weight_cap=10.0,
               dropout_rate=0.1,
               seed=None):
    super(TRPOActor, self).__init__()

    self.optimizer = optimizer
    # training parameters
    self.state_dim             = env.state_dim
    self.num_actions           = env.num_actions
    self.future_discount       = future_discount
    self.batch_size            = batch_size
    self.num_epoch             = num_epoch
    self.dropout_rate          = dropout_rate
    self.kl_targ               = kl_targ
    self.importance_weight_cap = importance_weight_cap
    self.env_sample            = env.env_sample
    self.output_types          = env.output_types
    self.output_shapes         = env.output_shapes
    self.beta                  = beta
    self.lam                   = lam
    self.seed                  = seed

    self.graph = tf.Graph()
    with self.graph.as_default():
      with tfv1.variable_scope(model_scope, default_name='trpo') as vs, tf.GradientTape(persistent=True) as gt:

        self.observations = tfv1.placeholder(
            self.output_types['observations'],
            self.output_shapes['observations'], name='observations')
        self.actions = tfv1.placeholder(
            self.output_types['actions'],
            self.output_shapes['actions'], name='actions')
        self.advantages = tfv1.placeholder(
            tf.dtypes.float32, [None], name='advantages')
        self.sampled_prob = tfv1.placeholder(
            tf.dtypes.float32, [None, self.num_actions], name='probs')
        self.dropout_pd = tfv1.placeholder(tf.float32, name='dropout_rate')
        self.state_visitation_frequency = tfv1.placeholder(
            tf.float32, name='state_visitation_frequency')

        if env.is_continuous:
          self.policy_network = ContinuousPolicyNN(
              self.num_actions, seed=seed)
          self.old_network = ContinuousPolicyNN(
              self.num_actions, seed=seed)
          self.prob_type = prob_type_lib.DiagGauss(self.num_actions)
          self.sampled_prob = tfv1.placeholder(
              tf.dtypes.float32, [None, self.num_actions * 2],
              name='probs')
        else:
          self.policy_network = DiscretePolicyNN(
              self.num_actions, seed=seed)
          self.old_network = DiscretePolicyNN(
              self.num_actions, seed=seed)
          self.prob_type = prob_type_lib.Categorical(self.num_actions)

        prob_pi, pi_var, prob_old = self.build_network(self.observations)
        self.prob_old = tf.stop_gradient(prob_old)
        prob_pi_t = self.build_loss(self.actions, prob_pi)
        train_op, pi_grad = self.build_opti_ops(prob_pi_t, pi_var, gt)
        sync_op = self.build_sync_network()

        # Keep handy fields.
        self.train_op = train_op
        self.gradients = pi_grad
        self.prob_old = prob_old
        self.prob_pi = prob_pi
        self.sync_op = sync_op

    self.sess = tfv1.Session(graph=self.graph, config=tfv1.ConfigProto(log_device_placement=True))

    # find memory footprint and compute cost of the model
    self.graph_size = utils_lib.graph_size(self.graph)
    with self.graph.as_default():
      self.sess.run(tfv1.global_variables_initializer())
      self.sess.run(tfv1.tables_initializer())

  def build_network(self, observations):
    # Define trainable variables.
    self.global_step = tf.Variable(0, name="global_step")
    prob = self.policy_network.forward(observations)
    prob_old = self.old_network.forward(observations)
    pi_var = self.policy_network.var()
    return prob, pi_var, prob_old

  def build_loss(self, actions, prob_pi):
    # Epsilon for numerical stability.
    eps = 1e-8
    # Get prediction on t.
    prob_pi_t = self.prob_type.likelihood(actions, prob_pi)
    # prob_old_t = self.prob_type.likelihood(actions, self.sampled_prob)
    prob_old_t = self.prob_type.likelihood(actions, self.prob_old)

    # Weight capping.
    importance_weight = tf.clip_by_value(
        utils_lib.stablize(prob_pi_t) / utils_lib.stablize(prob_old_t),
        clip_value_min=eps,
        clip_value_max=self.importance_weight_cap,
    )

    surr = tf.reduce_mean(importance_weight * self.advantages)
    # surr1 = importance_weight * self.advantages
    # surr2 = tf.clip_by_value(importance_weight, 0.9, 1.1) * self.advantages
    # surr = tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    # kl = self.prob_type.kl(self.sampled_prob, prob_pi)
    kl = tf.reduce_mean(
        self.prob_type.kl(self.prob_old, prob_pi)
    )
    # kl = self.prob_type.kl(tf.stop_gradient(prob_pi), prob_pi)
    kl_pen = self.beta * kl

    # Norm constraint.
    svf = self.state_visitation_frequency
    vargamma = tf.square(
        tf.norm(svf / utils_lib.stablize(tf.reduce_sum(svf)), ord=2))
    nc = vargamma * tf.math.abs(surr)

    self.surr = surr
    self.kl = kl
    self.importance_weight = importance_weight 

    return -(surr - kl_pen - nc)

  def build_opti_ops(self, prob_pi_t, pi_var, grad_tape):
    # compute gradients
    pi_grad = tf.gradients(prob_pi_t, pi_var)
    pi_grad = [(pi_grad[i], pi_var[i]) for i in range(len(pi_var))]
    pi_op = self.optimizer.apply_gradients(pi_grad)

    new_global_step = self.global_step + 1
    train_op = tf.group([pi_op, self.global_step.assign(new_global_step)])

    return train_op, pi_grad

  def build_sync_network(self):
    update_ops =[]
    new = self.policy_network.var()
    old = self.old_network.var()
    for i, n in enumerate(new):
      op = old[i].assign(n)
      update_ops.append(op)
    return update_ops

  def adapt_kl_penalty_coefficient(self, kl):
    if kl < self.kl_targ / 1.5:
      self.beta = self.beta / 2.0
    elif kl > self.kl_targ * 1.5:
      self.beta = self.beta * 2.0
    self.beta = min(self.beta, 1e20)
    self.beta = max(self.beta, 1e-20)

  def sync_old_policy(self):
    self.sess.run(self.sync_op)

  def sync_optimizer(self):
    if not hasattr(self.optimizer, 'set_params'):
      logging.error(self.optimizer)
      raise NotImplementedError
    self.optimizer.set_params(
        self.sess, self.get_params(self.optimizer.get_var_list()))

  def fit(self, steps, logger=None):
    kl_list = []
    iw_list = []
    self.num_timestep_seen += float(len(steps['observations']))
    # Train on data collected under old params.
    for e in range(self.num_epoch):
      steps = utils_lib.shuffle_map(steps)
      for i in range(len(steps['observations']))[::self.batch_size]:
        end = i + self.batch_size
        m = {
            self.observations: [step for step in steps['observations'][i:end]],
            self.actions: [step for step in steps['actions'][i:end]],
            self.advantages: [step for step in steps['advantages'][i:end]],
            self.sampled_prob: [step for step in steps['probs'][i:end]],
            self.dropout_pd: self.dropout_rate,
            self.state_visitation_frequency: steps['svf'],
        }

        # perform one update of training
        _, global_step, kl, iw, surr = self.sess.run([
            self.train_op,
            self.global_step,
            self.kl,
            self.importance_weight,
            self.surr,
          ], feed_dict=m
        )
        self.adapt_kl_penalty_coefficient(np.mean(kl))

        # Logging.
        # logging.error(m[self.advantages])
        # logging.error('%.20e, %.20e, %.20e' % (np.mean(kl), self.beta, np.mean(surr)))
        # logging.error("")
        kl_list.append(np.mean(kl))
        iw_list.append(np.mean(iw))

    # Iteration stat.
    return
    if logger:
      logger('# steps: %d' % len(steps['observations']))
      logger('Average kl in this iteration: %.20e' % (np.mean(kl_list)))
      logger('Average iw in this iteration: %.20e' % (np.mean(iw_list)))
      logger('Final beta in this iteration: %.20e' % (self.beta))

  def act(self, observations, stochastic=True):
    m = {
        self.observations: observations,
        self.actions: self.env_sample['actions'],
        self.dropout_pd: 0.0,
    }
    probs = self.sess.run(self.prob_pi, feed_dict=m)
    if stochastic:
      return self.prob_type.sample(probs), probs
    return self.prob_type.maxprob(probs), probs
