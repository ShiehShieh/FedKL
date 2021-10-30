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


class Params(object):

  def __init__(self, state_dim):
    self.lr = 1e-4
    self.top_k = 10
    self.future_discount = 0.99
    self.u_u = 512
    self.u_v = 256  # n_hidden.
    self.beta_unit_size = 256
    self.normalize_reward = False
    self.importance_weight_cap = 10.0
    self.dropout_rate = 0.0
    self.context_mode = 'LATENT_CROSS'


class Actor(ABC):
  @abstractmethod
  def build_network(self, observations, actions, next_observations,
                    seq_mask, delta_time, n_action, init_state,
                    mode, candidates, all_action_ids, context_mode):
    pass

  @abstractmethod
  def build_loss(self, reward, actions, logits_pi,
                 prob_pi, logits_beta, prob_beta):
    pass

  @abstractmethod
  def build_opti_ops(self, pi_var, loss_pi, beta_var,
                     loss_beta, grad_tape):
    pass


class TKPGActor(Actor):
  """docstring for TKPGActor"""

  def __init__(self, params):
    self.params = params

  def build_state_transition_network(self, observations,
                                     actions, next_observations,
                                     seq_mask, n_action,
                                     init_state, delta_time,
                                     mode, context_mode):
    state_size = self.params.u_v
    cfn_layer = tf.keras.layers.RNN(
        cfn_lib.CFNCell(state_size),
        return_sequences=True,
        return_state=True,
        name="pi_cfn")
    pi_u = tf.keras.layers.Embedding(
        n_action, self.params.u_u, name='action_u',
        embeddings_regularizer=tf.keras.regularizers.l2(0.1))
    u_a_t = pi_u(actions)
    pre_fusion = tf.keras.layers.Embedding(
        500, self.params.u_u, embeddings_initializer='normal',
        name='pre_fusion')
    post_fusion = tf.keras.layers.Embedding(
        500, state_size, embeddings_initializer='normal',
        name='post_fusion')
    context_features = {
        'LATENT_CROSS': lambda: tf.nn.relu((1.0 + pre_fusion(observations)) * u_a_t),
        'CONCATENATION': lambda: tf.concat([tf.cast(observations, tf.float32), u_a_t], axis=-1),
    }
    inputs = context_features[context_mode]()
    # Delta time pre-fusion by latent cross.
    dt_mat = tf.keras.layers.Dense(
        units=inputs.shape[-1], use_bias=False,
        name="dt_mat", kernel_initializer='glorot_normal')
    if delta_time is not None:
      # inputs = tf.nn.relu((1.0 + dt_mat(delta_time)) * inputs)
      inputs = (1.0 + dt_mat(delta_time)) * inputs
    if mode == ops_lib.Op.TRAINING and self.params.dropout_rate > 0.0:
      inputs = tf.keras.layers.Dropout(self.params.dropout_rate)(inputs)
    outputs, h_state = cfn_layer(
        inputs=inputs, mask=seq_mask,
        initial_state=[init_state])
    return cfn_layer, pi_u, outputs, h_state, pre_fusion, post_fusion, dt_mat

  def build_network(self, observations, actions,
                    next_observations, seq_mask,
                    delta_time, n_action, init_state,
                    mode, candidates, all_action_ids,
                    context_mode='LATENT_CROSS'):
    if context_mode not in ['CONCATENATION', 'LATENT_CROSS']:
      raise Exception('invalid context: %d' % context_mode)
    is_training = mode == ops_lib.Op.TRAINING
    # Define trainable variables.
    self.global_step = tf.Variable(0, name="global_step")
    # RNN can be masked out if seq_mask is all False. Do so in
    # case we don't want to use action_t, observation_t in the
    # very first recommendation. h_state will be zeros.
    cfn_layer, pi_u, user_state_t_1, h_state, pre_fusion, post_fusion, dt_mat = self.build_state_transition_network(
        observations, actions, next_observations, seq_mask,
        n_action, init_state, delta_time, mode, context_mode)
    v_size = {
        'LATENT_CROSS': self.params.u_v,
        'CONCATENATION': next_observations.shape[-1] + user_state_t_1.shape[-1],
    }[context_mode]
    pi_v = tf.keras.layers.Embedding(
        n_action, v_size, name='action_v',
        embeddings_regularizer=tf.keras.regularizers.l2(0.1))
    # Embedding shape: (vocab_size, embedding_dim)
    batch_size = tf.shape(actions)[0]
    batch_action_table = tf.tile(
        tf.expand_dims(tf.range(n_action), axis=0),
        multiples=[batch_size, 1])
    # (batch_size, n_action, u_v).
    v_embedding = pi_v(batch_action_table)
    # Beta.
    beta_d = tf.keras.layers.Dense(
        units=self.params.beta_unit_size,
        # activation=tf.keras.activations.tanh,
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        name="beta_d")
    beta_v = tf.keras.layers.Dense(
        units=n_action, use_bias=False, name="beta_v")

    # State size: (batch_size, frame_size, n_state + context_dim)
    observation_from_0_t_1 = tf.concat([tf.expand_dims(observations[:, 0], axis=1), next_observations], axis=1)
    state_from_0_t_1 = tf.concat([tf.expand_dims(init_state, 1), user_state_t_1], axis=1)
    context_features = {
        'LATENT_CROSS': lambda: (1.0 + post_fusion(observation_from_0_t_1)) * state_from_0_t_1,
        'CONCATENATION': lambda: tf.concat([tf.cast(observation_from_0_t_1, tf.float32), state_from_0_t_1], axis=-1),
    }
    user_state_from_0_t_1 = context_features[context_mode]()
    # No need to use remove_last_unmasked_step here. Both work though.
    # Unless we use layer normalization later on, it is vital to set unmasked
    # time step to zeros.
    user_state_t = utils_lib.remove_last_unmasked_step(
        user_state_from_0_t_1, seq_mask)
    context_features = {
        'LATENT_CROSS': lambda: tf.nn.relu(user_state_t),
        'CONCATENATION': lambda: user_state_t,
    }
    relu_state = context_features[context_mode]()

    # Estimate the target policy π(a|s), and keep numerically stable.
    pi_ln = tf.keras.layers.LayerNormalization(name='pi_ln')
    bd_ln = tf.keras.layers.LayerNormalization(name='bd_ln')
    bv_ln = tf.keras.layers.LayerNormalization(name='bv_ln')

    pi_var = cfn_layer.trainable_variables + pi_v.trainable_variables + pi_u.trainable_variables + pre_fusion.trainable_variables + post_fusion.trainable_variables + dt_mat.trainable_variables + pi_ln.trainable_variables
    beta_var = beta_v.trainable_variables + beta_d.trainable_variables + bd_ln.trainable_variables + bv_ln.trainable_variables

    # Populate the online inference results.
    # Not supporting batch prediction.
    seq_len_is_zero = tf.math.equal(
        tf.reduce_sum(tf.cast(seq_mask, tf.int32)), tf.constant(0, tf.int32))
    h_state = tf.cond(
        seq_len_is_zero, lambda: init_state, lambda: h_state)
    if mode == ops_lib.Op.SERVING:
      last_true_seq = utils_lib.argidx_last_unmasked(seq_mask)
      next_observation_last = tf.gather(next_observations, last_true_seq, batch_dims=1)
      # LATENT_CROSS: ndims == 2, CONCATENATION: ndims == 3.
      next_observation_last = tf.squeeze(next_observation_last, axis=1) if next_observation_last.get_shape().ndims == 3 else next_observation_last
      self.next_observation_last = next_observation_last
      context_features = {
          'LATENT_CROSS': lambda: tf.nn.relu(
              (1.0 + post_fusion(next_observation_last)) * h_state,
          ),
          'CONCATENATION': lambda: tf.expand_dims(
              tf.concat(
                  [tf.cast(next_observation_last, tf.float32), h_state],
                  axis=-1,
              ), axis=1,
          ),
      }
      # (batch_size, 1, v_size)
      f_relu_state = context_features[context_mode]()
      # Assuming that batch_size == 1 and seq_len == 1.
      if candidates is not None:
        # (n_candidate).
        candidates = tf.boolean_mask(
            candidates, tf.not_equal(candidates, -1))
        # tf.cond is expensive, so we insert a dummy id to avoid tf.cond.
        candidates = tf.concat(
            [candidates, tf.zeros(shape=[1], dtype=tf.int32)], axis=0)
        # (1, n_action, v_size) -> (1, n_candidates, v_size).
        v_embedding = tf.gather(v_embedding, candidates, axis=1)
        all_action_ids = tf.gather(all_action_ids, candidates)
      # Shape: (batch_size, 1, n_candidate).
      # logits_pi = tf.matmul(f_relu_state, v_a)
      logits_pi = tf.transpose(
          tf.matmul(
              v_embedding, tf.transpose(f_relu_state, perm=[0, 2, 1])),
          perm=[0, 2, 1])
      prob_pi = tf.nn.softmax(logits_pi)
      logits_beta = beta_v(
          tf.keras.layers.Activation('tanh')(
              beta_d(tf.stop_gradient(f_relu_state)))
      )
      prob_beta = tf.nn.softmax(logits_beta)
      return logits_pi, prob_pi, pi_var, logits_beta, prob_beta, beta_var, h_state, all_action_ids

    # (batch_size, u_v, n_action).
    v_a = tf.transpose(v_embedding, perm=[0, 2, 1])
    # Dropout on the output of RNN.
    if is_training and self.params.dropout_rate > 0.0:
      relu_state = tf.keras.layers.Dropout(
          self.params.dropout_rate)(relu_state)
    logits_pi = tf.matmul(relu_state, v_a)
    prob_pi = tf.nn.softmax(logits_pi)
    logits_beta = beta_v(
        tf.keras.layers.Activation('tanh')(
            beta_d(tf.stop_gradient(relu_state)))
    )
    prob_beta = tf.nn.softmax(logits_beta)

    self.user_state_t_1 = user_state_t_1
    self.user_state_t = user_state_t
    self.v_a = v_a
    self.init_state = init_state
    self.deb = logits_pi

    return logits_pi, prob_pi, pi_var, logits_beta, prob_beta, beta_var, h_state, all_action_ids

  def build_loss(self, reward, actions, seq_mask, n_action,
                 logits_pi, prob_pi, logits_beta, prob_beta):
    """Calculate losses for target and behavior policies.

    :reward [batch_size, T, 1]: The reward observed.
    :actions [batch_size, T, n_action]: The actions taken.

    """
    # Epsilon.
    eps = 1e-8
    # OneHot of actions.
    actions_oh = tf.one_hot(actions, depth=n_action, name='action_id_oh')
    batch_size, seq_len = tf.shape(actions)[0], tf.shape(actions)[1]
    actions_ep = tf.concat([tf.tile(
        tf.expand_dims(
            tf.expand_dims(tf.range(seq_len), axis=-1),
            axis=0,
        ),
        multiples=[batch_size, 1, 1],
    ), tf.expand_dims(actions, axis=-1)], axis=-1)
    # Get prediction on t.
    # prob_pi_t = tf.reduce_sum(prob_pi * actions_oh, axis=-1, keepdims=True)
    # prob_beta_t = tf.reduce_sum(prob_beta * actions_oh, axis=-1, keepdims=True)
    prob_pi_t = tf.gather_nd(prob_pi, actions_ep, batch_dims=1)
    prob_pi_t = tf.expand_dims(prob_pi_t, axis=-1)
    prob_beta_t = tf.gather_nd(prob_beta, actions_ep, batch_dims=1)
    prob_beta_t = tf.expand_dims(prob_beta_t, axis=-1)
    # Define α(a|s).
    lambda_k = self.params.top_k * (1 - prob_pi_t)**(self.params.top_k - 1)
    # Weight capping.
    importance_weight = tf.clip_by_value(
        prob_pi_t / prob_beta_t,
        clip_value_min=eps,
        clip_value_max=self.params.importance_weight_cap,
    )
    self.importance_weight = importance_weight
    # Top-K correction.
    top_k_factor = importance_weight * lambda_k
    self.top_k_factor = top_k_factor
    # Discounted future reward.
    # Reward in shape (batch_size, learning_timesteps, 1).
    dfr = tf.transpose(
        tf.scan(
            lambda a, x: self.params.future_discount * a + x,
            elems=tf.transpose(reward, perm=[1, 0, 2]),
            reverse=True),
        perm=[1, 0, 2],
    )
    # Normalize reward.
    if self.params.normalize_reward:
      seq_mask = tf.cast(seq_mask, dtype=tf.bool)
      masked_dfr = tf.boolean_mask(dfr, seq_mask)
      dfr = tf.subtract(dfr, tf.math.reduce_mean(masked_dfr))
      dfr = tf.math.divide_no_nan(dfr, tf.math.reduce_std(masked_dfr))
    # Normalize loss w.r.t. seq_mask.
    seq_mask = tf.cast(seq_mask, dtype=tf.float32)
    # Calculate Pi loss without importance sampling.
    raw_pi = dfr * tf.math.log(
        tf.maximum(prob_pi_t, eps),
    )
    raw_pi = tf.reduce_mean(
        -tf.reduce_sum(
            tf.squeeze(raw_pi, axis=2) * seq_mask, axis=1,
        ),
    )
    # Random predictor as baseline.
    raw_online = dfr * tf.math.log(1.0 / float(n_action))
    online_baseline = tf.reduce_mean(
        -tf.reduce_sum(
            tf.squeeze(raw_online, axis=2) * seq_mask, axis=1,
        ),
    )
    # Beta as baseline.
    raw_beta = dfr * tf.math.log(
        tf.maximum(prob_beta_t, eps),
    )
    beta_baseline = tf.reduce_mean(
        -tf.reduce_sum(
            tf.squeeze(raw_beta, axis=2) * seq_mask, axis=1,
        ),
    )
    # Calculate Pi loss.
    obj_pi = tf.stop_gradient(top_k_factor * dfr) * tf.math.log(tf.maximum(prob_pi_t, eps))
    obj_pi = tf.reduce_sum(
        tf.squeeze(obj_pi, axis=2) * seq_mask, axis=1,
    )
    pi = tf.reduce_mean(-obj_pi)
    pi = tf.where(tfv1.is_nan(pi), 0.69314694, pi)
    # Calculate Beta loss.
    seq_mask = tf.cast(seq_mask, dtype=tf.bool)
    obj_beta = tfv1.nn.softmax_cross_entropy_with_logits_v2(
        labels=actions_oh, logits=logits_beta)
    obj_beta = tf.boolean_mask(obj_beta, seq_mask)
    beta = tf.reduce_mean(obj_beta)
    beta = tf.where(tfv1.is_nan(beta), 0.69314694, beta)
    #
    self.obj_reward = tf.reduce_mean(tf.boolean_mask(tf.squeeze(dfr * tf.math.log(tf.maximum(prob_pi_t, eps))), seq_mask))
    self.obj_pi = prob_pi
    self.reward = reward
    self.dfr = dfr
    self.prob_pi_t = prob_pi_t

    return pi, beta, raw_pi, beta_baseline, online_baseline

  def build_opti_ops(self, pi_var, loss_pi, beta_var,
                     loss_beta, grad_tape):
    # Set optimizer for target policy π(a|s).
    logging.error(loss_pi)
    logging.error(pi_var)
    pi_grad = grad_tape.gradient(loss_pi, pi_var)
    pi_grad, _ = tf.clip_by_global_norm(
        pi_grad, clip_norm=1.0)
    pi = tf.optimizers.Adam(learning_rate=self.params.lr).apply_gradients(zip(pi_grad, pi_var))
    # Set optimizer for behavior policy beta(a|s).
    beta_grad = grad_tape.gradient(loss_beta, beta_var)
    beta_grad, _ = tf.clip_by_global_norm(
        beta_grad, clip_norm=1.0)
    beta = tf.optimizers.Adam(learning_rate=self.params.lr).apply_gradients(zip(beta_grad, beta_var))
    # Global step.
    new_global_step = self.global_step + 1
    train_op = tf.group([pi, beta, self.global_step.assign(new_global_step)])

    return train_op

  def build_metrics(self, actions, seq_mask,
                    beta_baseline, online_baseline,
                    logits_pi, prob_pi, raw_loss_pi,
                    logits_beta, prob_beta, loss_beta,
                    n_action, scope):
    ops = []  # Metrics update operations.
    inits = []  # Metrics initializers.
    # batch_size, seq_len = tf.shape(actions)[0], tf.shape(actions)[1]
    # Action coverage rate, Probability distribution.
    prob_pi_flat = tf.boolean_mask(prob_pi, seq_mask)
    pred_pi = tf.reduce_sum(
        tf.one_hot(
            tf.argmax(prob_pi_flat, axis=1), depth=n_action,
            dtype=tf.float32), axis=0)
    ac_preds = tf.Variable(
        tf.zeros((n_action), tf.float32), trainable=False)
    logging.error(ac_preds)
    acr = tf.reduce_mean(
        tf.cast(tf.cast(ac_preds, tf.bool), tf.float32))

    prob_pi_sum = tf.reduce_sum(prob_pi_flat, axis=0)
    pd_preds = tf.Variable(
        tf.zeros((n_action), tf.float32), trainable=False)
    pred_cnt = tf.Variable(
        tf.zeros((), tf.float32), trainable=False)
    num_pred = tf.cast(tf.shape(prob_pi_flat)[0], tf.float32)
    pd = tf.math.top_k(pd_preds / pred_cnt, k=2)
    ops.extend([
        ac_preds.assign_add(pred_pi),
        pd_preds.assign_add(prob_pi_sum),
        pred_cnt.assign_add(num_pred),
    ])
    inits.append(
        tfv1.variables_initializer(var_list=[
            ac_preds,
            pd_preds,
            pred_cnt,
        ]))
    if raw_loss_pi is not None and loss_beta is not None:
      # Average losses.
      pi_loss, pi_loss_op = tfv1.metrics.mean(
          raw_loss_pi, name="loss_metrics")
      beta_loss, beta_loss_op = tfv1.metrics.mean(
          loss_beta, name="loss_metrics")
      beta_baseline, beta_baseline_op = tfv1.metrics.mean(
          beta_baseline, name="loss_metrics")
      online_baseline, online_baseline_op = tfv1.metrics.mean(
          online_baseline, name="loss_metrics")
      ops.extend([pi_loss_op, beta_loss_op, beta_baseline_op,
                  online_baseline_op])
      inits.append(
          tfv1.variables_initializer(
              var_list=tfv1.get_collection(
                  tfv1.GraphKeys.LOCAL_VARIABLES,
                  scope="%s/loss_metrics" % scope.name)))
      logging.error(tfv1.get_collection(
                  tfv1.GraphKeys.LOCAL_VARIABLES,
                  scope="%s/loss_metrics" % scope.name))
      # Beta as Baseline.
      # Trajectory as Baseline.
    # Pi vs. Beta.
    actions_flat = tf.boolean_mask(actions, seq_mask)
    label_flat = tf.cast(actions_flat, tf.float32)
    # Consistency of Pi policy w.r.t. online Beta policy.
    prob_pi_flat = tf.boolean_mask(prob_pi, seq_mask)
    pi_rec, pi_rec_opt = tfv1.metrics.accuracy(
        label_flat, tf.argmax(prob_pi_flat, axis=1),
        name='pi_rec_metric')
    inits.append(
        tfv1.variables_initializer(
            var_list=tfv1.get_collection(
                tfv1.GraphKeys.LOCAL_VARIABLES,
                scope="%s/pi_rec_metric" % scope.name)))
    logging.error(tfv1.get_collection(
                tfv1.GraphKeys.LOCAL_VARIABLES,
                scope="%s/pi_rec_metric" % scope.name))
    # Overall accuracy of Beta policy.
    prob_beta_flat = tf.boolean_mask(prob_beta, seq_mask)
    beta_acc, beta_acc_opt = tfv1.metrics.accuracy(
        label_flat, tf.argmax(prob_beta_flat, axis=1),
        name='beta_acc_metric')
    inits.append(
        tfv1.variables_initializer(
            var_list=tfv1.get_collection(
                tfv1.GraphKeys.LOCAL_VARIABLES,
                scope="%s/beta_acc_metric" % scope.name)))
    logging.error(tfv1.get_collection(
                tfv1.GraphKeys.LOCAL_VARIABLES,
                scope="%s/beta_acc_metric" % scope.name))
    # Mismatch between Pi and example trajectory.
    argmax_prob_pi = tf.argmax(
        prob_pi, axis=2, output_type=tf.int32)
    pi_diff_action = tf.cast(
        tf.equal(actions, argmax_prob_pi), tf.int32)

    ops.extend([pi_rec_opt, beta_acc_opt])
    met_op = tf.group(ops)
    met_init_op = tf.group(inits)

    return met_op, met_init_op, acr, pd, pi_rec, pi_loss, beta_acc, beta_loss, beta_baseline, online_baseline, pi_diff_action

  def build_graph(self, inputs, n_action, mode):
    # inputs = {
    #     'state_embedding': inputs[0],
    #     'action_id': inputs[1],
    #     'reward': inputs[2],
    #     'next_state_embedding': inputs[3],
    #     'seq_mask': inputs[4],
    #     'label_ctx': inputs[5],
    #     'hidden_state': inputs[6],
    # }
    observations = inputs['state_embedding']
    next_observations = inputs['next_state_embedding']
    seq_mask = inputs['seq_mask']
    actions = inputs['action_id']
    init_state = inputs['hidden_state']
    reward = inputs['reward']
    all_action_ids = inputs['all_action_ids']
    candidates = None
    delta_time = None
    if 'delta_time' in inputs:
      delta_time = inputs['delta_time']
    if 'user_id' in inputs:
      self.user_id = inputs['user_id']
    if 'debug' in inputs:
      self.deb = inputs['debug']
    if 'candidates' in inputs and mode == ops_lib.Op.SERVING:
      candidates = inputs['candidates']
      self.deb = candidates

    self.observations = observations
    self.actions = actions
    self.next_observations = next_observations
    self.seq_mask = seq_mask
    with tfv1.variable_scope(None, default_name='tkpg') as vs, tf.GradientTape(persistent=True) as gt:
      logits_pi, prob_pi, pi_var, logits_beta, prob_beta, beta_var, h_state, all_action_ids = self.build_network(
          observations, actions, next_observations, seq_mask,
          delta_time, n_action, init_state, mode,
          candidates, all_action_ids, self.params.context_mode)

      self.logits_pi = logits_pi
      self.prob_pi = prob_pi
      self.pi_var = pi_var
      self.logits_beta = logits_beta
      self.prob_beta = prob_beta
      self.beta_var = beta_var
      self.h_state = h_state
      self.seq_mask = seq_mask
      self.all_action_ids = all_action_ids

      if mode == ops_lib.Op.SERVING:
        return

      loss_pi, loss_beta, raw_loss_pi, beta_baseline, online_baseline = self.build_loss(
          reward, actions, seq_mask, n_action, logits_pi,
          prob_pi, logits_beta, prob_beta)
      if mode == ops_lib.Op.TRAINING:
        train_op = self.build_opti_ops(
            pi_var, loss_pi, beta_var, loss_beta, gt)
        self.train_op = train_op

      # Using raw loss pi which ignores importance weight.
      met_op, met_init, acr, pd, pi_rec, pi_loss_met, beta_acc, beta_loss_met, beta_baseline_met, online_baseline_met, pi_diff_action = self.build_metrics(
          actions, seq_mask, beta_baseline, online_baseline,
          logits_pi, prob_pi, raw_loss_pi,
          logits_beta, prob_beta, loss_beta, n_action, vs)

      self.loss_pi = loss_pi
      self.loss_beta = loss_beta
      self.raw_loss_pi = raw_loss_pi
      self.met_op = met_op
      self.met_init_op = met_init
      self.acr = acr
      self.pd = pd
      self.pi_rec = pi_rec
      self.pi_loss_met = pi_loss_met
      self.beta_acc = beta_acc
      self.beta_loss_met = beta_loss_met
      self.beta_baseline_met = beta_baseline_met
      self.online_baseline_met = online_baseline_met
      self.pi_diff_action = pi_diff_action
