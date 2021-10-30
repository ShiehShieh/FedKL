from __future__ import absolute_import, division, print_function

import tensorflow as tf

# from rl.tkpg.py.util.dict_util import WriteOnceObject
from model.utils.dict_util import WriteOnceObject

unknown = ""
na = "n/a"


def build_model_constants(mc):
  """Build model constants.

    Args:
      mc: ModelConstants proto.

    Returns:
      Nested WriteOnceObject containing the tensors built from model constants.
    """
  output = WriteOnceObject()

  output.action_kinds = _build_table_object(
      mc.all_action_kind[:], set_attr=True)
  output.obs_kinds = _build_table_object(
      mc.all_observation_kind[:], set_attr=True)
  output.dimensions = _build_table_object(mc.all_dimension[:], set_attr=True)
  output.activity_types = _build_table_object(mc.all_activity_type[:])

  a_keys = [a.activity_key for a in mc.activity]
  output.activities = _build_table_object(a_keys, default=-1)
  output.activities.mask = tf.constant([1] * (output.activities.cnt), tf.int32)
  # a_keys = [unknown] + [a.activity_key for a in mc.activity]
  # output.activities = _build_table_object(a_keys, default=0)
  # output.activities.unknown = 0
  # output.activities.mask = tf.constant([0] + [1] * (output.activities.cnt - 1), tf.int32)

  # Prepare activity infos.
  # output.activities = _build_table_object(
  #     [unknown] + mc.all_activity_key[:], default=0)
  # output.activities.unknown = 0

  # Prepare KP infos.
  # output.kps = _build_table_object([unknown] + mc.all_kp_key[:], default=0)
  # output.kps.unknown = 0
  output.kps = _build_table_object(mc.all_kp_key[:], default=-1)
  # output.show_kps = _build_table_object(
  #     [unknown] + mc.all_show_kp_key[:], default=0)
  # output.show_kps.unknown = 0
  output.show_kps = _build_table_object(mc.all_show_kp_key[:], default=-1)

  # Activity show KP.
  tl = [tf.zeros([output.show_kps.cnt], dtype=tf.int32)]
  for l in mc.activity:
    sk_keys = tf.constant([v for v in l.show_kp_key], dtype=tf.string)
    sk_mh = tf.reduce_max(output.show_kps.one_hot(sk_keys), axis=0)
    tl.append(sk_mh)
  output.activities.sk_mh = tf.stack(tl, axis=0)
  output.activities.sk_cnt = tf.reduce_sum(output.activities.sk_mh, axis=-1)

  return output


def _build_tf_table(keys, values, dtype=tf.float32, default=0):
  return tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys, values, value_dtype=dtype),
      default)


def _build_table_object(keys, default=-1, set_attr=False):
  obj = WriteOnceObject()
  obj.cnt = len(keys)
  obj.keys = tf.constant(keys)
  obj.table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          [na] + keys, [-1] + list(range(obj.cnt)), value_dtype=tf.int32), default)
  # default = 0
  # obj.table = tf.lookup.StaticHashTable(
  #     tf.lookup.KeyValueTensorInitializer(
  #         [na] + keys, list(range(obj.cnt + 1)), value_dtype=tf.int32), default)
  if set_attr:
    for i, v in enumerate(keys):
      setattr(obj, v.lower(), i)

  def _one_hot(xs, dtype=tf.int32):
    return tf.one_hot(obj.table.lookup(xs), depth=obj.cnt, dtype=dtype)

  obj.one_hot = _one_hot
  return obj
