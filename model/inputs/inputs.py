from __future__ import absolute_import, division, print_function

# from absl import logging

import tensorflow as tf

import model.rl.agent.ops as ops_lib
import model.utils.utils as utils_lib
# from rl.tkpg.py.util.dict_util import WriteOnceObject
# import rl.tkpg.py.model.model_util as mlib

na = "n/a"


def _fixed_len_col(dtype=tf.int64, shape=()):
  return tf.io.FixedLenFeature(shape, dtype=dtype)


def _var_seq_col(dtype=tf.int64, shape=()):
  return tf.io.FixedLenSequenceFeature(shape=shape,
                                       dtype=dtype,
                                       allow_missing=True)


def _var_len_col(dtype=tf.int64):
  return tf.io.VarLenFeature(dtype=dtype)


def _ragged_col(value_key="", dtype=tf.float32):
  return tf.io.RaggedFeature(value_key=value_key, dtype=dtype)


_COLUMN_SPECS = {
    # Contexts.
    "user_id": _fixed_len_col(),
    "seq_len": _fixed_len_col(),
    # RNN states.
    "hidden_state": _var_len_col(tf.float32),

    # Sequence meta.
    "t": _var_seq_col(),
    "idx": _var_seq_col(),
    "candidates": _ragged_col("candidates", tf.string),

    # Actions.
    "a/kind": _ragged_col("a/kind", tf.string),
    "a/action_id": _ragged_col("a/action_id", tf.string),
    "a/timestamp_usec": _ragged_col("a/timestamp_usec", tf.int64),

    # Observations.
    "o/kind": _ragged_col("o/kind", tf.string),
    "o/kp_key": _ragged_col("o/kp_key", tf.string),
    "o/kp/score": _ragged_col("o/kp/score"),
    "o/sk_key": _ragged_col("o/sk_key", tf.string),
    "o/sk/prof_avg": _ragged_col("o/sk/prof_avg"),
    "o/sk/prof_begin": _ragged_col("o/sk/prof_begin"),
    "o/sk/prof_end": _ragged_col("o/sk/prof_end"),
    # "o/activity_key": _ragged_col("o/activity_key", tf.string),
    # "o/activity/score": _ragged_col("o/activity/score"),
    # Mastered show KP key.
    "o/gp_sk_key": _ragged_col("o/gp_sk_key", tf.string),
    # Proficiency info.
    "o/step_sk_key": _ragged_col("o/step_sk_key", tf.string),
    "o/step_sk_prof": _ragged_col("o/step_sk_prof", tf.float32),
}


def _get_column_spec(cols):
  return {col: _COLUMN_SPECS[col] for col in cols}


def skv_line_decoder():
  """Decode a single simple kv line.

  Args:
    ln: string, simple kv line.

  Returns:
    A callable to be passed to dataset map function.
  """

  def _decode(ln):
    skv_val = tf.compat.v1.string_split(tf.expand_dims(ln, axis=0),
                                        "\t").values[-1]
    serialized = tf.io.decode_base64(skv_val)
    return serialized

  return _decode


def sequence_example_decoder(ctx_cols, ftr_cols):
  """Decode TF sequence examples.

  Args:
    ln: serialized tf.SequenceExample proto.
    ctx_cols: context columns to decode.
    ftr_cols: sequence feature columns to decode.

  Returns:
    A callable to be passed to dataset map function.
  """

  # Take the value part of the simple kv line.
  def _decode(ln):
    return tf.io.parse_sequence_example(
        ln,
        context_features=_get_column_spec(ctx_cols),
        sequence_features=_get_column_spec(ftr_cols))

  return _decode


def build_serving_inputs(mc, n_hidden, serialized, ctx_cols, ftr_cols,
                         max_seq_len):
  """Build serving inputs builds inputs tuple from a serialized tf example.

  Args:
    mc: nested WriteOnceObject containing model constants.
    serialized: String tensor containing serialized tf sequence example.
    ctx_cols: context columns to decode.
    ftr_cols: sequence feature columns to decode.

  Returns:
    Nested WriteOnceObject containing input tensors as well as derived tensors.
  """
  ctx, ftr, _ = tf.io.parse_sequence_example(
      serialized,
      context_features=_get_column_spec(ctx_cols),
      sequence_features=_get_column_spec(ftr_cols))
  ftr.update(ctx)
  return build_inputs_from_dict(mc, n_hidden, ftr, max_seq_len,
                                ops_lib.Op.SERVING)


def build_inputs_from_dict(mc, n_hidden, kvs, max_seq_len, op):
  """
  Build input tensors from dictionary of tensors.

  Returns:
    Nested WriteOnceObject containing input tensors as well as derived tensors.
  """
  # timestamp must be int64.
  # # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # # So cast all int64 to int32.
  # for name in list(kvs.keys()):
  #   t = kvs[name]
  #   if t.dtype == tf.int64:
  #     t = tf.cast(t, tf.int32)
  #   kvs[name] = t

  # Prepare timestep index and seqence mask inputs.
  seq_len = kvs["seq_len"] - 1  # See remove_xxx_step.
  padded_seq_len = tf.reduce_max(seq_len)
  seq_len_is_zero = tf.math.equal(padded_seq_len, 0)
  padded_seq_len = tf.cond(seq_len_is_zero, lambda: tf.constant(1, tf.int64),
                           lambda: padded_seq_len)
  seq_mask = tf.sequence_mask(seq_len, padded_seq_len, dtype=tf.int32)
  seq_mask = seq_mask[:, :max_seq_len]

  # action_kinds = kvs["a/kind"].to_tensor(default_value=na)
  # # action_kinds_idx = mc.action_kinds.table.lookup(action_kinds)
  # action_kinds_oh = mc.action_kinds.one_hot(action_kinds)
  # action_kinds_oh = tf.concat([action_kinds_oh], axis=-1)

  a_keys = kvs["a/action_id"].to_tensor(default_value=na)
  a_keys = a_keys[:, :max_seq_len + 1]
  a_keys_idx = mc.activities.table.lookup(a_keys)
  a_keys_idx = a_keys_idx[:, :, 0]

  a_ts = kvs["a/timestamp_usec"].to_tensor(default_value=0)
  a_ts = a_ts[:, :max_seq_len + 1]
  a_ts = a_ts[:, :, 0]

  # Populate observations of each time step.
  obs_kinds = kvs["o/kind"].to_tensor(default_value=na)
  obs_kinds = obs_kinds[:, :max_seq_len + 1]
  # obs_kinds_idx = mc.obs_kinds.table.lookup(obs_kinds)
  obs_kinds_oh = mc.obs_kinds.one_hot(obs_kinds, tf.float32)
  # # Assuming that there is only one observation per step.
  # obs_kinds_oh = tf.reduce_max(obs_kinds_oh, axis=2)
  # obs_kinds_oh = tf.gather(obs_kinds_oh, 0, batch_dims=2)
  obs_kinds_oh = obs_kinds_oh[:, :, 0]

  o_tensors = [obs_kinds_oh]
  if "o/kp_key" in kvs:
    kp_keys = kvs["o/kp_key"].to_tensor(default_value=na)
    kp_keys = kp_keys[:, :max_seq_len + 1]
    # kp_keys_idx = mc.kps.table.lookup(kp_keys)
    kp_keys_oh = mc.kps.one_hot(kp_keys, tf.float32)
    kp_score = kvs["o/kp/score"].to_tensor()
    kp_score = kp_score[:, :max_seq_len + 1]
    kp_score = tf.expand_dims(kp_score, axis=-1)
    # kp_score = tf.tile(tf.expand_dims(kp_score, axis=-1),
    #                    multiples=[1, 1, 1, kp_keys_oh.shape[-1]])
    kp_score_mh = tf.reduce_max(kp_keys_oh * kp_score, axis=2)
    o_tensors.append(kp_score_mh)

  if "o/sk_key" in kvs:
    sk_keys = kvs["o/sk_key"].to_tensor(default_value=na)
    sk_keys = sk_keys[:, :max_seq_len + 1]
    # sk_keys_idx = mc.show_kps.table.lookup(sk_keys)
    sk_keys_oh = mc.show_kps.one_hot(sk_keys, tf.float32)

    sk_prof_avg = kvs["o/sk/prof_avg"].to_tensor()
    sk_prof_avg = sk_prof_avg[:, :max_seq_len + 1]
    sk_prof_avg = tf.expand_dims(sk_prof_avg, axis=-1)
    # sk_prof_avg = tf.tile(
    #     tf.expand_dims(sk_prof_avg, axis=-1),
    #     multiples=[1, 1, 1, sk_keys_oh.shape[-1]])
    sk_prof_avg_mh = tf.reduce_max(sk_keys_oh * sk_prof_avg, axis=2)
    o_tensors.append(sk_prof_avg_mh)

    sk_prof_begin = kvs["o/sk/prof_begin"].to_tensor()
    sk_prof_begin = sk_prof_begin[:, :max_seq_len + 1]
    sk_prof_begin = tf.expand_dims(sk_prof_begin, axis=-1)
    # sk_prof_begin = tf.tile(
    #     tf.expand_dims(sk_prof_begin, axis=-1),
    #     multiples=[1, 1, 1, sk_keys_oh.shape[-1]])
    sk_prof_begin_mh = tf.reduce_max(sk_keys_oh * sk_prof_begin, axis=2)
    o_tensors.append(sk_prof_begin_mh)

    sk_prof_end = kvs["o/sk/prof_end"].to_tensor()
    sk_prof_end = sk_prof_end[:, :max_seq_len + 1]
    sk_prof_end = tf.expand_dims(sk_prof_end, axis=-1)
    # sk_prof_end = tf.tile(
    #     tf.expand_dims(sk_prof_end, axis=-1),
    #     multiples=[1, 1, 1, sk_keys_oh.shape[-1]])
    sk_prof_end_mh = tf.reduce_max(sk_keys_oh * sk_prof_end, axis=2)
    o_tensors.append(sk_prof_end_mh)

  # Populate observations of history state in each step.
  if "o/gp_sk_key" in kvs:
    gp_sk_keys = kvs["o/gp_sk_key"].to_tensor(default_value=na)
    gp_sk_keys = gp_sk_keys[:, :max_seq_len + 1]
    # gp_sk_keys_idx = mc.show_kps.table.lookup(gp_sk_keys)
    gp_sk_oh = mc.show_kps.one_hot(gp_sk_keys, tf.float32)
    gp_sk_mh = tf.reduce_max(gp_sk_oh, axis=2)
    o_tensors.append(gp_sk_mh)

  if "o/step_sk_key" in kvs:
    sk_keys = kvs["o/step_sk_key"].to_tensor(default_value=na)
    sk_keys = sk_keys[:, :max_seq_len + 1]
    sk_oh = mc.show_kps.one_hot(sk_keys, tf.float32)
    sk_prof = kvs["o/step_sk_prof"].to_tensor()
    sk_prof = sk_prof[:, :max_seq_len + 1]
    sk_prof = tf.expand_dims(sk_prof, axis=-1)
    # sk_prof = tf.tile(tf.expand_dims(sk_prof, axis=-1),
    #                   multiples=[1, 1, 1, sk_oh.shape[-1]])
    sk_prof_mh = tf.reduce_max(sk_oh * sk_prof, axis=2)
    o_tensors.append(sk_prof_mh)
  deb = sk_prof_mh

  safe_remove = op == ops_lib.Op.SERVING
  safe_remove = safe_remove

  def compute_reward(sk_prof_mh, seq_mask):
    prof_t_1 = utils_lib.remove_first_step(sk_prof_mh)
    # The last unmasked step will affect the calculation of the discounted
    # future reward.
    prof_t = utils_lib.remove_last_unmasked_step(sk_prof_mh, seq_mask)
    residual = prof_t_1 - prof_t
    return tf.reduce_sum(residual, axis=2, keepdims=True)

  # Observations, actions and rewards.
  obs = tf.concat(o_tensors, axis=-1)
  # o_t = utils_lib.remove_last_unmasked_step(obs, seq_mask)
  # a_t = utils_lib.remove_first_step(a_keys_idx)
  # o_t_1 = utils_lib.remove_first_step(obs)
  # r = compute_reward(sk_prof_mh, seq_mask)
  o_t, a_t, o_t_1, r = tf.cond(
      seq_len_is_zero,
      lambda: [
          obs,
          a_keys_idx,  # Must be -1 (n/a).
          obs,
          tf.zeros((tf.shape(obs)[0], tf.shape(obs)[1], 1)),
      ],
      lambda: [
          utils_lib.remove_last_unmasked_step(obs, seq_mask),
          utils_lib.remove_first_step(a_keys_idx),
          utils_lib.remove_first_step(obs),
          compute_reward(sk_prof_mh, seq_mask),
      ],
  )
  batch_size, seq_len = tf.shape(a_t)[0], tf.shape(a_t)[1]
  # Delta time.
  # It is ok to use ts_now, cauze we will drop the last RNN
  # state anyway during training.
  day_in_usec = 86400000000
  ts_now = tf.cast(tf.timestamp(name='pad_timestamp_now') * 1e6, tf.int64)
  # ts_t = utils_lib.remove_first_step(a_ts)
  # ts_t_1 = utils_lib.remove_first_step(
  #     utils_lib.append_last_unmasked_step(
  #         ts_t, seq_mask, ts_now))
  ts_t = tf.cond(
      seq_len_is_zero,
      lambda: a_ts,
      lambda: utils_lib.remove_first_step(a_ts),
  )
  ts_t_1 = tf.cond(
      seq_len_is_zero,
      lambda: a_ts,
      lambda: utils_lib.remove_first_step(
          utils_lib.append_last_unmasked_step(ts_t, seq_mask, ts_now)),
  )
  dt_t = tf.expand_dims(tf.math.log(
      tf.cast(tf.math.floor((ts_t_1 - ts_t) / day_in_usec) + 1, tf.float32)),
                        axis=2)
  # Mask out time step in which action == -1. The impact on transition function
  # is likely to be trivial, since they're minor.
  unknown_mask = tf.cast(tf.equal(a_t, -1), tf.int32)
  seq_mask = tf.cast(seq_mask - (seq_mask * unknown_mask), tf.bool)
  a_t = a_t + unknown_mask  # -1 to 0. Or one_hot complains.

  hs = tf.zeros([batch_size, n_hidden], dtype=tf.float32)
  if "hidden_state" in kvs:
    # Check length of hidden state vector.
    kvs["hidden_state"] = tf.sparse.to_dense(kvs["hidden_state"])
    hs_shape = hs.shape.as_list()
    hs = tf.cond(
        tf.equal(tf.shape(kvs["hidden_state"])[-1], n_hidden),
        lambda: kvs["hidden_state"],
        lambda: hs,
    )
    hs.set_shape(hs_shape)
  hidden_state = hs
  cands = tf.fill([batch_size, seq_len, 0], -1)
  if "candidates" in kvs:
    cands = kvs["candidates"].to_tensor(default_value=na)
    cands = cands[:, :max_seq_len + 1]
    cands = mc.activities.table.lookup(cands)
    cands = tf.cond(
        seq_len_is_zero,
        lambda: cands,
        lambda: utils_lib.remove_first_step(cands),
    )
  candidates = cands

  # After course observations.
  user_id = tf.constant(0, dtype=tf.int64)
  if "user_id" in kvs:
    user_id = kvs["user_id"]
  return {
      'user_id': user_id,
      'state_embedding': o_t,
      'next_state_embedding': o_t_1,
      'seq_mask': seq_mask,
      'action_id': a_t,
      'hidden_state': hidden_state,
      'reward': r,
      'delta_time': dt_t,
      'candidates': candidates,
      'all_action_ids': mc.activities.keys,
      'debug': deb,
  }
