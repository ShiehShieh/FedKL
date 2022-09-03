#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from absl import logging
import numpy as np

from collections import deque

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
  def __init__(
      self, num_actions, dropout_rate=None, seed=None, linear=False):
    self.dropout_rate = dropout_rate
    self.seed = seed
    self.linear = linear
    # # Reacher.
    # self.h1 = tf.keras.layers.Dense(
    #     units=64, use_bias=True,
    #     activation='tanh',
    #     kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    #     kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
    #     name="h1")
    # self.h2 = tf.keras.layers.Dense(
    #     units=64, use_bias=True,
    #     activation='tanh',
    #     kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    #     kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
    #     name="h2")
    # For Flow SUMO.
    self.h1 = tf.keras.layers.Dense(
        units=100, use_bias=True,
        activation='tanh',
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
        name="h1")
    self.h2 = tf.keras.layers.Dense(
        units=50, use_bias=True,
        activation='tanh',
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
        name="h2")
    self.h3 = tf.keras.layers.Dense(
        units=25, use_bias=True,
        activation='tanh',
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
        name="h3")
    self.o = tf.keras.layers.Dense(
        units=num_actions, use_bias=True,
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed),
        name="policy_network_o")
    self.r = tf.Variable(
        shape=(num_actions,), dtype=tf.float32, trainable=True,
        initial_value=[0.0] * num_actions, name='logstd')

  def var(self):
    if self.linear:
      return self.o.trainable_variables + [self.r]
    # return self.h1.trainable_variables + self.h2.trainable_variables + self.o.trainable_variables + [self.r]
    return self.h1.trainable_variables + self.h2.trainable_variables + self.h3.trainable_variables + self.o.trainable_variables + [self.r]

  def forward(self, observations):
    if self.linear:
      mean = self.o(observations)
    else:
      # mean = self.o(self.h2(self.h1(observations)))
      mean = self.o(self.h3(self.h2(self.h1(observations))))
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
               nm_targ=0.01,
               nm_targ_adap=(1.0, 0.1, 50.0),
               lam=0.98,
               beta=1.0,
               sigma=0.0,
               mu=0.0,
               fixed_sigma=False,
               importance_weight_cap=10.0,
               dropout_rate=0.1,
               distance_metric='sqrt_kl',
               gradient_clip_norm=None,
               seed=None,
               linear=False,
               verbose=True):
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
    self.nm_targ               = nm_targ
    self.nm_targ_adap          = nm_targ_adap
    self.importance_weight_cap = importance_weight_cap
    self.env_sample            = env.env_sample
    self.output_types          = env.output_types
    self.output_shapes         = env.output_shapes
    self.lam                   = lam
    self.seed                  = seed
    self.distance_metric       = distance_metric
    self.avg_kl                = 0.0
    self.avg_nm                = 0.0
    self.lst_kl                = 0.0
    self.lst_nm                = 0.0
    self.max_kl                = 0.0
    self.max_nm                = 0.0
    self.max_ad                = 0.0
    self.avg_ad                = 0.0
    self.is_norm_penalized     = True
    self.is_proximal           = True
    self.fixed_sigma           = fixed_sigma
    self.log_sg                = deque(maxlen=10)
    self.log_nm                = deque(maxlen=10)
    self.log_kl                = deque(maxlen=10)
    self.log_sr                = deque(maxlen=10)
    self.init_sigma            = sigma
    self.gradient_clip_norm    = gradient_clip_norm

    if sigma == 0.0:
      self.is_norm_penalized = False
    if mu == 0.0:
      self.is_proximal = False

    self.graph = tf.Graph()
    with self.graph.as_default():
      self.beta = tf.Variable(initial_value=beta)
      self.sigma = tf.Variable(initial_value=sigma)
      self.mu = tf.Variable(initial_value=mu)

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
        self.norm_penalty = tfv1.placeholder(
            tf.float32, name='norm_penalty')

        if env.is_continuous:
          self.policy_network = ContinuousPolicyNN(
              self.num_actions, seed=seed, linear=linear)
          self.old_network = ContinuousPolicyNN(
              self.num_actions, seed=seed, linear=linear)
          self.anchor_network = ContinuousPolicyNN(
              self.num_actions, seed=seed, linear=linear)
          # self.backup_network = ContinuousPolicyNN(
          #     self.num_actions, seed=seed, linear=linear)
          self.prob_type = prob_type_lib.DiagGauss(self.num_actions)
          self.sampled_prob = tfv1.placeholder(
              tf.dtypes.float32, [None, self.num_actions * 2],
              name='probs')
        else:
          self.policy_network = DiscretePolicyNN(
              self.num_actions, seed=seed)
          self.old_network = DiscretePolicyNN(
              self.num_actions, seed=seed)
          self.anchor_network = DiscretePolicyNN(
              self.num_actions, seed=seed)
          # self.backup_network = DiscretePolicyNN(
          #     self.num_actions, seed=seed)
          self.prob_type = prob_type_lib.Categorical(self.num_actions)

        # prob_pi, pi_var, prob_old, prob_anchor, prob_backup = self.build_network(
        #     self.observations)
        prob_pi, pi_var, prob_old, prob_anchor = self.build_network(
            self.observations)
        prob_pi_t = self.build_loss(
            self.actions, prob_pi, prob_old, prob_anchor)
        train_op, pi_grad, grad_norm = self.build_opti_ops(
            prob_pi_t, pi_var, gt)
        # sync_old_op, sync_anchor_op, sync_backup_op, restore_backup_op = self.build_network_sync()
        sync_old_op, sync_anchor_op = self.build_network_sync()

        # Keep handy fields.
        self.train_op = train_op
        self.gradients = pi_grad
        self.prob_old = prob_old
        self.prob_anchor = prob_anchor
        self.prob_pi = prob_pi
        self.sync_old_op = sync_old_op
        self.sync_anchor_op = sync_anchor_op
        # self.sync_backup_op = sync_backup_op
        # self.restore_backup_op = restore_backup_op
        self.grad_norm = grad_norm

    self.sess = tfv1.Session(graph=self.graph, config=tfv1.ConfigProto(log_device_placement=verbose))

    # find memory footprint and compute cost of the model
    self.graph_size = utils_lib.graph_size(self.graph)
    with self.graph.as_default():
      self.sess.run(tfv1.global_variables_initializer())
      self.sess.run(tfv1.tables_initializer())

  def build_network(self, observations):
    # Define trainable variables.
    self.global_step = tf.Variable(0, name="global_step")
    prob = self.policy_network.forward(observations)
    old = tf.stop_gradient(self.old_network.forward(observations))
    anchor = tf.stop_gradient(self.anchor_network.forward(observations))
    # backup = tf.stop_gradient(self.backup_network.forward(observations))
    pi_var = self.policy_network.var()
    # return prob, pi_var, old, anchor, backup
    return prob, pi_var, old, anchor

  def build_loss(self, actions, prob_pi, prob_old, prob_anchor):
    # Epsilon for numerical stability.
    eps = 1e-8
    # Get prediction on t.
    prob_pi_t = self.prob_type.likelihood(actions, prob_pi)
    # prob_old_t = self.prob_type.likelihood(actions, self.sampled_prob)
    prob_old_t = self.prob_type.likelihood(actions, prob_old)
    prob_anchor_t = self.prob_type.likelihood(actions, prob_anchor)

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
        self.prob_type.kl(prob_old, prob_pi)
    )
    # kl = self.prob_type.kl(tf.stop_gradient(prob_pi), prob_pi)
    kl_pen = self.beta * kl

    # Norm constraint.
    svf = self.state_visitation_frequency
    # NOTE(XIE.Zhijie): Assuming that the current update will not affect other
    # states.
    #
    # TODO(XIE,Zhijie): This advantage is not of \pi^t yet.
    distance_metric = None
    if self.distance_metric == 'mahalanobis':
      distance_metric = self.prob_type.mahalanobis
    elif self.distance_metric == 'wasserstein':
      distance_metric = self.prob_type.wasserstein
    elif self.distance_metric == 'tv':
      distance_metric = self.prob_type.tv
    elif self.distance_metric == 'kl':
      distance_metric = self.prob_type.kl
    elif self.distance_metric == 'sqrt_kl':
      distance_metric = lambda x, y: tf.sqrt(
          tf.maximum(self.prob_type.kl(x, y), 1e-8)
      )
    else:
      raise NotImplementedError
    # nm = self.norm_penalty * tf.reduce_mean(distance_metric(
    #     prob_anchor, prob_pi) * tf.math.abs(self.advantages))
    nm = tf.reduce_mean(distance_metric(prob_anchor, prob_pi))
    nm_pen = self.sigma * nm
    if not self.is_norm_penalized:
      nm_pen = 0.0
    prox = self.proximal_term(self.anchor_network, self.policy_network)
    px_pen = self.mu / 2.0 * prox
    if not self.is_proximal:
      px_pen = 0.0

    self.surr = surr
    self.kl = kl
    self.importance_weight = importance_weight
    self.nm = nm
    self.prox = prox

    return -(surr - kl_pen - nm_pen) + px_pen

  def build_opti_ops(self, prob_pi_t, pi_var, grad_tape):
    # compute gradients
    pi_grad = tf.gradients(prob_pi_t, pi_var)
    grad_norm = tf.linalg.global_norm(pi_grad)
    if self.gradient_clip_norm is not None:
      pi_grad, _ = tf.clip_by_global_norm(
          pi_grad, clip_norm=self.gradient_clip_norm, use_norm=grad_norm)
    pi_grad = [(pi_grad[i], pi_var[i]) for i in range(len(pi_var))]
    pi_op = self.optimizer.apply_gradients(pi_grad)

    new_global_step = self.global_step + 1
    train_op = tf.group([pi_op, self.global_step.assign(new_global_step)])

    return train_op, pi_grad, grad_norm

  def build_network_sync(self):
    # For old network.
    update_ops_old =[]
    new = self.policy_network.var()
    old = self.old_network.var()
    for i, n in enumerate(new):
      op = old[i].assign(n)
      update_ops_old.append(op)
    # For anchor network.
    update_ops_anchor =[]
    new = self.policy_network.var()
    anchor = self.anchor_network.var()
    for i, n in enumerate(new):
      op = anchor[i].assign(n)
      update_ops_anchor.append(op)
    # # For backup network.
    # update_ops_backup =[]
    # new = self.policy_network.var()
    # backup = self.backup_network.var()
    # for i, n in enumerate(new):
    #   op = backup[i].assign(n)
    #   update_ops_backup.append(op)
    # # For restoring backup network.
    # update_ops_restore =[]
    # new = self.policy_network.var()
    # backup = self.backup_network.var()
    # for i, n in enumerate(backup):
    #   op = new[i].assign(n)
    #   update_ops_restore.append(op)
    # return update_ops_old, update_ops_anchor, update_ops_backup, update_ops_restore
    return update_ops_old, update_ops_anchor

  def proximal_term(self, n1, n2):
    v1 = n1.var()
    v2 = n2.var()
    norm = tf.zeros(())
    for i, v in enumerate(v1):
      # norm = norm + tf.square(tf.norm(v1[i] - v2[i], ord='2'))
      # norm = norm + tf.reduce_sum(tf.square(v1[i] - v2[i]))
      norm = norm + tf.nn.l2_loss(v1[i] - v2[i])
    return norm

  def adapt_kl_penalty_coefficient(self, kl, scale):
    beta = self.sess.run(self.beta)
    if beta == 0.0:
      return beta
    # These hyper-parameters depend on the magnitude of target KL.
    threshold = 1.5
    threshold = 1.1
    step = 5.0
    step = 2.0
    if kl < self.kl_targ / threshold:
      beta = beta / step
    elif kl > self.kl_targ * threshold:
      beta = beta * step
    beta = max(beta, 1e-8)
    beta = min(min(beta, scale / kl), 20.0)
    self.beta.load(beta, self.sess)
    return beta

  def adapt_nm_penalty_coefficient(self, nc, scale):
    sigma = self.sess.run(self.sigma)
    if sigma == 0.0 or self.fixed_sigma:
      return sigma
    if not self.is_norm_penalized:
      return sigma
    self.log_sg.append(sigma)
    threshold = 1.5
    threshold = 1.1
    step = 5.0
    step = 2.0
    # step = 10.0
    if nc < self.nm_targ / threshold:
      sigma = sigma / step
    elif nc > self.nm_targ * threshold:
      sigma = sigma * step
    sigma = max(sigma, self.init_sigma)
    sigma = min(min(sigma, scale / nc), 20.0)
    self.sigma.load(sigma, self.sess)
    return sigma

  def set_nm_penalty_coefficient(self, v):
    self.sigma.load(v, self.sess)

  def adapt_nm_target(self, num):
    s, e, span = self.nm_targ_adap
    i = (e - s) / span * num
    if s > e:
      self.nm_targ = max(s + i, e)
    else:
      self.nm_targ = min(s + i, e)

  def set_nm_targ(self, targ):
    self.nm_targ = targ

  def sync_old_policy(self):
    self.sess.run(self.sync_old_op)

  def sync_anchor_policy(self):
    self.sess.run(self.sync_anchor_op)

  # def sync_backup_policy(self):
  #   self.sess.run(self.sync_backup_op)

  # def restore_backup_policy(self):
  #   self.sess.run(self.restore_backup_op)

  def sync_optimizer(self):
    if not hasattr(self.optimizer, 'set_params'):
      logging.error(self.optimizer)
      raise NotImplementedError
    self.optimizer.set_params(
        self.sess, self.get_params(self.optimizer.get_var_list()))

  def reset_optimizer(self):
    if not hasattr(self.optimizer, 'reset_params'):
      logging.error(self.optimizer)
      raise NotImplementedError
    self.optimizer.reset_params(self.sess)

  def stat(self):
    beta, sigma = self.sess.run([self.beta, self.sigma])
    return {
        'beta': beta,
        'sigma': sigma,
        'avg_kl': self.avg_kl,
        'avg_nm': self.avg_nm,
        'avg_sr': self.avg_sr,
        'lst_kl': self.lst_kl,
        'lst_nm': self.lst_nm,
        'lst_sr': self.lst_sr,
        'max_kl': self.max_kl,
        'max_nm': self.max_nm,
        'max_sr': self.max_sr,
        'log_sg': self.log_sg,
        'max_ad': self.max_ad,
        'max_gn': self.max_gn,
        'min_gn': self.min_gn,
        'avg_gn': self.avg_gn,
        # 'avg_ad': self.avg_ad,
    }

  def fit(self, steps, logger=None):
    gn_list = []
    self.num_fit += 1.0
    self.num_timestep_seen += float(len(steps['observations']))
    scale = np.max(steps['advantages'])
    # If nan occurs, we can restore to this point.
    # self.sync_backup_policy()
    # Train on data collected under old params.
    for e in range(self.num_epoch):
      steps = utils_lib.shuffle_map(steps)
      for i in range(len(steps['observations']))[::self.batch_size]:
        end = i + self.batch_size
        m = {
            self.observations: [step for step in steps['observations'][i:end]],
            self.actions: [step for step in steps['actions'][i:end]],
            self.advantages: [step for step in steps['advantages'][i:end]],
            # self.sampled_prob: [step for step in steps['probs'][i:end]],
            self.dropout_pd: self.dropout_rate,
            self.state_visitation_frequency: steps['svf'],
            self.norm_penalty: steps['norm_penalty'],
        }

        # Perform one batch of training.
        _, global_step, kl, nm, gn = self.sess.run([
            self.train_op,
            self.global_step,
            self.kl,
            self.nm,
            self.grad_norm,
            # self.surr,
          ], feed_dict=m
        )
        gn_list.append(gn)
        # if np.any([np.isnan(s) for s in [kl, nm, gn, sr]]):
        #   # Stop the training and restore to the checkpoint.
        #   logging.error('restoring. %s' % ([kl, nm, gn, sr]))
        #   self.restore_backup_policy()
        #   return True

    # Adaptively change penalty coefficients.
    m = {
        self.observations: steps['observations'],
        self.actions: steps['actions'],
        self.advantages: steps['advantages'],
    }
    kl, nm, sr = self.sess.run([
        self.kl,
        self.nm,
        self.surr,
      ], feed_dict=m
    )
    beta = self.adapt_kl_penalty_coefficient(kl, scale)
    sigma = self.adapt_nm_penalty_coefficient(nm, scale)
    self.log_kl.append(kl)
    self.log_nm.append(nm)
    self.log_sr.append(sr)

    # Iteration stat.
    self.avg_kl = np.mean(self.log_kl)
    self.avg_nm = np.mean(self.log_nm)
    self.avg_sr = np.mean(self.log_sr)
    self.lst_kl = kl
    self.lst_nm = nm
    self.lst_sr = sr
    self.max_kl = np.max(self.log_kl)
    self.max_nm = np.max(self.log_nm)
    self.max_sr = np.max(self.log_sr)
    self.max_ad = np.max(steps['advantages'])
    self.avg_ad = np.mean(steps['advantages'])
    self.max_gn = np.max(gn_list)
    self.min_gn = np.min(gn_list)
    self.avg_gn = np.mean(gn_list)

    if sigma > 100.0:
      return False
    return True

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
