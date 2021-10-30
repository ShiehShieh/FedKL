#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from absl import logging
from abc import ABC, abstractmethod
import random
import numpy as np

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import model.rl.agent.pg as pg_lib
import model.utils.utils as utils_lib

tfv1.disable_eager_execution()

tfv1.disable_v2_behavior()
# Need to enable TFv2 control flow with support for higher order derivatives
# in keras LSTM layer.
tfv1.enable_control_flow_v2()


class PolicyNN(object):
  def __init__(self, num_actions):
    self.l = tf.keras.layers.Dense(
        units=100,
        # activation=tf.keras.activations.tanh,
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        name="policy_network_l")
    self.o = tf.keras.layers.Dense(
        units=num_actions, use_bias=False, name="policy_network_o")

  def forward(self, observations, actions):
    logits = self.o(
        tf.keras.layers.Activation('tanh')(
            self.l(observations))
    )
    return logits


class REINFORCEActor(pg_lib.PolicyGradient):
  def __init__(self,
               env,
               optimizer,
               model_scope,
               batch_size=16,
               init_exp=0.5,
               final_exp=0.0,
               anneal_steps=500,
               future_discount=0.99,
               normalize_reward=False,
               importance_weight_cap=10.0,
               dropout_rate=0.0):
    self.optimizer = optimizer
    # training parameters
    self.state_dim             = env.state_dim
    self.num_actions           = env.num_actions
    self.future_discount       = future_discount
    self.batch_size            = batch_size
    self.epsilon               = init_exp
    self.init_exp              = init_exp
    self.final_exp             = final_exp
    self.anneal_steps          = anneal_steps
    self.dropout_rate          = dropout_rate
    self.normalize_reward      = normalize_reward
    self.importance_weight_cap = importance_weight_cap
    self.env_sample            = env.env_sample
    self.output_types          = env.output_types
    self.output_shapes         = env.output_shapes

    self.graph = tf.Graph()
    with self.graph.as_default():
      with tfv1.variable_scope(model_scope, default_name='reinforce') as vs, tf.GradientTape(persistent=True) as gt:

        # def f():
        #   for i in self.channel:
        #     yield i
        #   raise tf.errors.OutOfRangeError
        # dataset = tf.data.Dataset.from_generator(
        #     f,
        #     output_types=self.output_types,
        #     output_shapes=self.output_shapes,
        # )
        # train_dataset = dataset.repeat(
        #     1).shuffle(
        #         1).batch(
        #             self.batch_size)  # .prefetch(tf.data.experimental.AUTOTUNE)
        # iterator = tfv1.data.make_initializable_iterator(train_dataset)
        # inputs = iterator.get_next()
        # logging.error(inputs)

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

    self.sess = tfv1.Session(graph=self.graph)

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
    # OneHot of actions.
    actions_oh = tf.one_hot(actions, depth=self.num_actions, name='action_id_oh')
    batch_size, seq_len = tf.shape(actions)[0], tf.shape(actions)[1]
    actions_ep = tf.concat([tf.tile(
        tf.expand_dims(
            tf.expand_dims(tf.range(seq_len), axis=-1),
            axis=0,
        ),
        multiples=[batch_size, 1, 1],
    ), tf.expand_dims(actions, axis=-1)], axis=-1)
    # Get prediction on t.
    prob_pi_t = tf.gather_nd(prob_pi, actions_ep, batch_dims=1)
    prob_pi_t = tf.expand_dims(prob_pi_t, axis=-1)

    return prob_pi_t

    # Weight capping.
    importance_weight = tf.clip_by_value(
        prob_pi_t / 1.0,
        clip_value_min=eps,
        clip_value_max=self.importance_weight_cap,
    )
    # Discounted future reward.
    # Reward in shape (batch_size, learning_timesteps, 1).
    dfr = tf.transpose(
        tf.scan(
            lambda a, x: self.future_discount * a + x,
            elems=tf.transpose(reward, perm=[1, 0, 2]),
            reverse=True),
        perm=[1, 0, 2],
    )
    # Normalize reward.
    if self.normalize_reward:
      seq_mask = tf.cast(seq_mask, dtype=tf.bool)
      masked_dfr = tf.boolean_mask(dfr, seq_mask)
      dfr = tf.subtract(dfr, tf.math.reduce_mean(masked_dfr))
      dfr = tf.math.divide_no_nan(dfr, tf.math.reduce_std(masked_dfr))
    # Normalize loss w.r.t. seq_mask.
    seq_mask = tf.cast(seq_mask, dtype=tf.float32)
    # # Calculate Pi loss.
    # obj_pi = tf.stop_gradient(dfr) * tf.math.log(tf.maximum(prob_pi_t, eps))
    # obj_pi = tf.reduce_sum(
    #     tf.squeeze(obj_pi, axis=2) * seq_mask, axis=1,
    # )
    # loss_pi = tf.reduce_mean(-obj_pi)
    # loss_pi = tf.where(tfv1.is_nan(loss_pi), 0.69314694, loss_pi)

  def build_opti_ops(self, prob_pi_t, pi_var, grad_tape):
    # compute gradients
    pi_grad = tf.gradients(prob_pi_t, pi_var, grad_ys=tf.expand_dims(-self.dfr, axis=2))
    pi_grad = [(pi_grad[i], pi_var[i]) for i in range(len(pi_var))]
    pi_op = self.optimizer.apply_gradients(pi_grad)

    new_global_step = self.global_step + 1
    train_op = tf.group([pi_op, self.global_step.assign(new_global_step)])

    return train_op, pi_grad

  def anneal_exploration(self, global_step):
    ratio = max((self.anneal_steps - global_step) / float(self.anneal_steps), 0)
    self.epsilon = (self.init_exp - self.final_exp) * ratio + self.final_exp

  def fit(self, steps):
    for k, v in steps.items():
      steps[k] = list(reversed(v.tolist()))
    for i in range(len(steps))[::self.batch_size]:
      end = i + self.batch_size
      m = {
          self.observations: [[step] for step in steps['observations'][i:end]],
          self.actions: [[step] for step in steps['actions'][i:end]],
          self.reward: [[[step]] for step in steps['reward'][i:end]],
          self.dfr: [[step] for step in steps['dfr'][i:end]],
          self.seq_mask: [[1] for step in steps['dfr'][i:end]],
      }

      # perform one update of training
      _, global_step = self.sess.run([
          self.train_op,
          self.global_step,
        ], feed_dict=m
      )
      # TODO: Timestep. No right.
      self.anneal_exploration(global_step)

  def act(self, observations):
    # epsilon-greedy exploration strategy
    if random.random() < self.epsilon:
      return random.randint(0, self.num_actions - 1)
    else:
      m = {
          self.observations: [[observations]],
          self.actions: self.env_sample['actions'],
          self.reward: self.env_sample['reward'],
          self.seq_mask: self.env_sample['seq_mask'],
          self.dfr: self.env_sample['dfr'],
      }
      probs = self.sess.run(self.prob_pi, feed_dict=m)[0, 0]
      return np.argmax(np.random.multinomial(1, probs))






#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from absl import logging
from abc import ABC, abstractmethod
import random
import numpy as np

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import model.rl.agent.pg as pg_lib
import model.utils.utils as utils_lib

tfv1.disable_eager_execution()

tfv1.disable_v2_behavior()
# Need to enable TFv2 control flow with support for higher order derivatives
# in keras LSTM layer.
tfv1.enable_control_flow_v2()


class PolicyNN(object):
  def __init__(self, num_actions):
    self.l = tf.keras.layers.Dense(
        units=100,
        # activation=tf.keras.activations.tanh,
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        name="policy_network_l")
    self.o = tf.keras.layers.Dense(
        units=num_actions, use_bias=False, name="policy_network_o")

  def forward(self, observations, actions):
    logits = self.o(
        tf.keras.layers.Activation('tanh')(
            self.l(observations))
    )
    return logits


class REINFORCEActor(pg_lib.PolicyGradient):
  def __init__(self,
               env,
               optimizer,
               model_scope,
               init_exp=0.5,
               final_exp=0.0,
               anneal_steps=500,
               future_discount=0.99,
               normalize_reward=False,
               importance_weight_cap=10.0,
               dropout_rate=0.0,
               summary_writer=None,
               summary_every=100):
    self.optimizer = optimizer
    # training parameters
    self.state_dim             = env.state_dim
    self.num_actions           = env.num_actions
    self.future_discount       = future_discount
    self.epsilon               = init_exp
    self.init_exp              = init_exp
    self.final_exp             = final_exp
    self.anneal_steps          = anneal_steps
    self.dropout_rate          = dropout_rate
    self.normalize_reward      = normalize_reward
    self.importance_weight_cap = importance_weight_cap
    self.env_sample            = env.env_sample
    self.output_types          = env.output_types
    self.output_shapes         = env.output_shapes
    self.channel               = []

    self.graph = tf.Graph()
    with self.graph.as_default():
      with tfv1.variable_scope(model_scope, default_name='reinforce') as vs, tf.GradientTape(persistent=True) as gt:

        def f():
          for i in self.channel:
            yield i
          raise tf.errors.OutOfRangeError
        dataset = tf.data.Dataset.from_generator(
            f,
            output_types=self.output_types,
            output_shapes=self.output_shapes,
        )
        train_dataset = dataset.repeat(
            1).shuffle(
                1).batch(
                    1)  # .prefetch(tf.data.experimental.AUTOTUNE)
        iterator = tfv1.data.make_initializable_iterator(train_dataset)
        inputs = iterator.get_next()
        logging.error(inputs)

        self.observations = inputs['observations']
        self.actions = inputs['actions']
        self.seq_mask = inputs['seq_mask']
        self.reward = inputs['reward']
        self.dfr = inputs['dfr']

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

        self.build_summary()

    self.sess = tfv1.Session(graph=self.graph)

    # find memory footprint and compute cost of the model
    self.graph_size = utils_lib.graph_size(self.graph)
    with self.graph.as_default():
      self.sess.run(tfv1.global_variables_initializer())
      self.sess.run(tfv1.tables_initializer())
      self.sess.run(iterator.initializer)
      # metadata = tfv1.RunMetadata()
      # opts = tfv1.profiler.ProfileOptionBuilder.float_operation()
      # self.flops = tfv1.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

  def feed_dataset_to_channel(self, step):
    self.channel.append(step)

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
    # OneHot of actions.
    actions_oh = tf.one_hot(actions, depth=self.num_actions, name='action_id_oh')
    batch_size, seq_len = tf.shape(actions)[0], tf.shape(actions)[1]
    actions_ep = tf.concat([tf.tile(
        tf.expand_dims(
            tf.expand_dims(tf.range(seq_len), axis=-1),
            axis=0,
        ),
        multiples=[batch_size, 1, 1],
    ), tf.expand_dims(actions, axis=-1)], axis=-1)
    # Get prediction on t.
    prob_pi_t = tf.gather_nd(prob_pi, actions_ep, batch_dims=1)
    prob_pi_t = tf.expand_dims(prob_pi_t, axis=-1)
    # Weight capping.
    importance_weight = tf.clip_by_value(
        prob_pi_t / 1.0,
        clip_value_min=eps,
        clip_value_max=self.importance_weight_cap,
    )
    # Discounted future reward.
    # Reward in shape (batch_size, learning_timesteps, 1).
    dfr = tf.transpose(
        tf.scan(
            lambda a, x: self.future_discount * a + x,
            elems=tf.transpose(reward, perm=[1, 0, 2]),
            reverse=True),
        perm=[1, 0, 2],
    )
    # Normalize reward.
    if self.normalize_reward:
      seq_mask = tf.cast(seq_mask, dtype=tf.bool)
      masked_dfr = tf.boolean_mask(dfr, seq_mask)
      dfr = tf.subtract(dfr, tf.math.reduce_mean(masked_dfr))
      dfr = tf.math.divide_no_nan(dfr, tf.math.reduce_std(masked_dfr))
    # Normalize loss w.r.t. seq_mask.
    seq_mask = tf.cast(seq_mask, dtype=tf.float32)
    # # Calculate Pi loss.
    # obj_pi = tf.stop_gradient(dfr) * tf.math.log(tf.maximum(prob_pi_t, eps))
    # obj_pi = tf.reduce_sum(
    #     tf.squeeze(obj_pi, axis=2) * seq_mask, axis=1,
    # )
    # loss_pi = tf.reduce_mean(-obj_pi)
    # loss_pi = tf.where(tfv1.is_nan(loss_pi), 0.69314694, loss_pi)

    return prob_pi_t

  def build_opti_ops(self, prob_pi_t, pi_var, grad_tape):
    # compute gradients
    pi_grad = grad_tape.gradient(prob_pi_t, pi_var)
    pi_grad = [(pi_grad[i], pi_var[i]) for i in range(len(pi_var))]

    # compute policy gradients
    self.dfr = tf.gather(self.dfr, 0)
    for i, (grad, var) in enumerate(pi_grad):
      if grad is not None:
        pi_grad[i] = (-grad * self.dfr, var)  # Maximizing.
        logging.error('%d, %s' % (i, var))

    logging.error(pi_var)
    pi_op = self.optimizer.apply_gradients(pi_grad)

    new_global_step = self.global_step + 1
    train_op = tf.group([pi_op, self.global_step.assign(new_global_step)])

    return train_op, pi_grad

  def build_summary(self):
    self.no_op = tf.no_op()
    return
    for grad, var in self.gradients:
      tfv1.summary.histogram(var.name, var)
      if grad is not None:
        tfv1.summary.histogram(var.name + '/gradients', grad)

    # emit summaries
    self.summarize = tfv1.summary.merge_all()

  def anneal_exploration(self, global_step):
    ratio = max((self.anneal_steps - global_step) / float(self.anneal_steps), 0)
    self.epsilon = (self.init_exp - self.final_exp) * ratio + self.final_exp

  def fit(self, trajectories):
    for trajectory in trajectories:
      dfr = 0.0
      for step in reversed(trajectory):
        dfr = self.future_discount * dfr + step['reward']
        m = {
            "observations": [step['observations']],
            "actions": [step['actions']],
            "reward": [[step['reward']]],
            "dfr": [dfr],
            # "seq_mask": tf.sequence_mask(len(trajectory), len(trajectory), dtype=tf.int32),
            "seq_mask": [1],
        }
        self.feed_dataset_to_channel(m)

        calculate_summaries = False
        # perform one update of training
        _, summary_str, global_step = self.sess.run([
            self.train_op,
            self.summarize if calculate_summaries else self.no_op,
            self.global_step,
          ]
        )
        self.anneal_exploration(global_step)

  def act(self, observations):
    # epsilon-greedy exploration strategy
    if random.random() < self.epsilon:
      return random.randint(0, self.num_actions - 1)
    else:
      m = {
          "observations": [observations],
          "actions": self.env_sample['actions'],
          "reward": self.env_sample['reward'],
          "seq_mask": self.env_sample['seq_mask'],
          "dfr": self.env_sample['seq_mask'],
      }
      self.feed_dataset_to_channel(m)
      probs = self.sess.run(self.prob_pi)[0, 0]
      return np.argmax(np.random.multinomial(1, probs))
