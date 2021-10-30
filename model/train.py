from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
import copy
import glob
import gzip
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import os
import random

from protos.activity import model_constants_pb2 as mcpb
import model.inputs.inputs as inputs_lib
import model.inputs.model_constants as mc_lib
import model.rl.agent.agent as agent_lib
import model.rl.agent.ops as ops_lib
from time import time

# Set a seed value
seed_value = 12321

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
# import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
# import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
# import numpy as np
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
# import tensorflow as tf
tfv1.set_random_seed(seed_value)

FLAGS = flags.FLAGS

flags.DEFINE_string("trajectory_files", None,
                    "Glob patterns for trajectory simple kv gzip.")
flags.DEFINE_string("model_constants_pb", None, "Model constants protobuf.")
flags.DEFINE_string("op", None, "Operation, can be one defined in Op")
flags.DEFINE_string("tf_log_basedir", None,
                    "Basedir to be read by tensorboard.")
flags.DEFINE_integer("batch_size", 32, "Sample size for one batch.")
flags.DEFINE_integer("num_epoches", 1, "Maximum number of training epoches.")
flags.DEFINE_integer("log_freq", 100,
                     "Logging frequency of batches during training.")
flags.DEFINE_integer("max_train_steps", None,
                     "If not None, run at most this many steps.")
flags.DEFINE_string("export_basedir", None,
                    "Path to export model to, used if --op=EXPORT_SERVING.")
flags.DEFINE_string("run_id", None,
                    "A string suggesting what this run is about.")
flags.DEFINE_integer("seed", 17, "Random seed.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length of each trajectory. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


def _get_training_dir():
  return os.path.join(FLAGS.tf_log_basedir, FLAGS.run_id, "TRAINING")


def get_gzip_line_num(files):
  out = 0
  for f in files:
    with gzip.GzipFile(f) as a:
      for line in a:
        out += 1
  return out


def split_train_test(ds, dataset_size, train_ratio=0.8):
  train_size = int(train_ratio * dataset_size)
  train_dataset = ds.take(train_size)
  test_dataset = ds.skip(train_size)
  return train_dataset, test_dataset


def copy_model_parameters(sess, scope1, scope2):
    """
    Copies the model parameters of one estimator to another.
    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = {t.name.lstrip(scope1): t for t in tfv1.trainable_variables() if t.name.startswith(scope1) and not t.name.startswith(scope2)}
    e2_params = {t.name.lstrip(scope2): t for t in tfv1.trainable_variables() if t.name.startswith(scope2)}

    update_ops = []
    for e2_k, e2_v in e2_params.items():
      e1_v = e1_params.get(e2_k, None)
      if e1_v is None:
        logging.fatal('model 2 must be a subset of model 1:\nmodel 1: %s\nmodel 2: %s.' % (e1_params, e2_params))
      op = e2_v.assign(e1_v)
      update_ops.append(op)

    sess.run(update_ops)


def _build_graph(op, mc, n_action, files=None):
  ctx_cols = [
      "user_id",
      "seq_len",
      # "order",
  ]
  ftr_cols = [
      "t",
      "idx",
      "a/kind",
      "a/action_id",
      "a/timestamp_usec",
      "o/kind",
      # "o/kp_key",
      # "o/kp/score",
      "o/sk_key",
      "o/sk/prof_avg",
      "o/sk/prof_begin",
      "o/sk/prof_end",
      # "o/activity_key",
      # "o/activity/score",
      "o/gp_sk_key",
      "o/step_sk_key",
      "o/step_sk_prof",
  ]

  agent = agent_lib.Agent(agent_lib.Params(n_action, op))
  agent.params.actor_params.top_k = 1
  agent.params.actor_params.lr = 1.0e-04
  agent.params.actor_params.future_discount = 0.1
  agent.params.actor_params.u_u = 256
  agent.params.actor_params.u_v = 256
  agent.params.actor_params.beta_unit_size = 256
  agent.params.actor_params.normalize_reward = False
  agent.params.actor_params.importance_weight_cap = 1.0
  agent.params.actor_params.dropout_rate = 0.2
  agent.params.actor_params.context_mode = 'CONCATENATION'

  val_agent = copy.deepcopy(agent)
  val_agent.params.mode = ops_lib.Op.TESTING

  def filter_fn(x, y, z):
    logging.error(x['sample_wise'])
    return True

  train_dataset, test_dataset = None, None
  if op in (ops_lib.Op.TRAINING, ops_lib.Op.TESTING):
    ds = tf.data.TextLineDataset(
        files, compression_type="GZIP")
    dataset_size = get_gzip_line_num(files)
    train_dataset, test_dataset = split_train_test(
        ds, dataset_size)
    logging.error(dataset_size)
  if op == ops_lib.Op.TRAINING:
    train_dataset = train_dataset.repeat(
        FLAGS.num_epoches).map(
            inputs_lib.skv_line_decoder()).shuffle(
                FLAGS.batch_size * 100).batch(
                    FLAGS.batch_size).map(
                        inputs_lib.sequence_example_decoder(
                            ctx_cols, ftr_cols),
                        num_parallel_calls=20).prefetch(
                            tf.data.experimental.AUTOTUNE)
    logging.error(train_dataset)

    it = tfv1.data.make_one_shot_iterator(train_dataset)
    kvs = it.get_next()
    kvs[1].update(kvs[0])
    inputs = inputs_lib.build_inputs_from_dict(
        mc, agent.params.actor_params.u_v, kvs[1],
        FLAGS.max_seq_length, op)
    logging.error(inputs)

    test_dataset = test_dataset.repeat(1).map(
        inputs_lib.skv_line_decoder()).batch(1).map(
            inputs_lib.sequence_example_decoder(
                ctx_cols, ftr_cols),
            num_parallel_calls=20).prefetch(
                tf.data.experimental.AUTOTUNE)
    logging.error(test_dataset)
    val_it = tfv1.data.make_initializable_iterator(test_dataset)
    kvs = val_it.get_next()
    kvs[1].update(kvs[0])
    val_inputs = inputs_lib.build_inputs_from_dict(
        mc, agent.params.actor_params.u_v, kvs[1],
        FLAGS.max_seq_length, op)
    logging.error(val_inputs)

    return agent.build_graph(inputs), val_agent.build_graph(val_inputs), val_it
  elif op == ops_lib.Op.TESTING:
    test_dataset = test_dataset.repeat(1).map(
        inputs_lib.skv_line_decoder()).batch(1).map(
            inputs_lib.sequence_example_decoder(ctx_cols, ftr_cols),
            num_parallel_calls=20)

    logging.error(test_dataset)
    it = tfv1.data.make_one_shot_iterator(test_dataset)
    kvs = it.get_next()
    kvs[1].update(kvs[0])
    inputs = inputs_lib.build_inputs_from_dict(
        mc, agent.params.actor_params.u_v, kvs[1],
        FLAGS.max_seq_length, op)
    logging.error(inputs)
    return agent.build_graph(inputs)
  elif op == ops_lib.Op.EXPORT_SERVING:
    agent.params.mode = ops_lib.Op.SERVING
    # Extend features for online inference.
    ctx_cols.extend([
        "hidden_state",
    ])
    ftr_cols.extend([
        "candidates",
    ])
    serving_input = tfv1.placeholder(
        shape=(1,), dtype=tf.string, name="serialized_tf_example")
    inputs = inputs_lib.build_serving_inputs(
        mc, agent.params.actor_params.u_v, serving_input,
        ctx_cols, ftr_cols, FLAGS.max_seq_length)
    return serving_input, agent.build_graph(inputs)
  return


def _run_training(tensors, val_tensors, val_it, max_train_steps, log_freq=100):
  logging.info("Training TKPG model...")
  basedir = _get_training_dir()

  # Training.
  train_op = tensors['train_op']
  global_step_tensor = tensors['global_step']
  loss_pi_tensor = tensors['loss_pi']
  loss_beta_tensor = tensors['loss_beta']
  raw_loss_pi_tensor = tensors['raw_loss_pi']
  iw_tensor = tensors['importance_weight']
  # Metrics.
  met_op_tensor = tensors['met_op']
  met_init_op_tensor = tensors['met_init_op']
  acr_tensor = tensors['acr']
  pd_tensor = tensors['pd']
  pi_rec_tensor = tensors['pi_rec']
  pi_loss_met_tensor = tensors['pi_loss_met']
  beta_acc_tensor = tensors['beta_acc']
  beta_loss_met_tensor = tensors['beta_loss_met']

  val_met_op_tensor = val_tensors['met_op']
  val_met_init_op_tensor = val_tensors['met_init_op']
  val_acr_tensor = val_tensors['acr']
  val_pd_tensor = val_tensors['pd']
  val_pi_rec_tensor = val_tensors['pi_rec']
  val_pi_loss_met_tensor = val_tensors['pi_loss_met']
  val_beta_acc_tensor = val_tensors['beta_acc']
  val_beta_loss_met_tensor = val_tensors['beta_loss_met']
  val_beta_baseline_met_tensor = val_tensors['beta_baseline_met']
  val_online_baseline_met_tensor = val_tensors['online_baseline_met']

  batch_time = 0
  global_step = 0
  deb = None
  loss_pi, loss_beta, raw_loss_pi, iw = 0.0, 0.0, 0.0, 0.0

  with tfv1.Session() as sess:
    # Initialize all the variables.
    sess.run(tfv1.global_variables_initializer())
    sess.run(tfv1.tables_initializer())
    sess.run(met_init_op_tensor)
    # Saver.
    saver = tfv1.train.Saver(tfv1.trainable_variables(), max_to_keep=2)
    for batch_idx in itertools.count():
      try:
        # deb = sess.run([
        #     tensors['deb'],
        # ])
        # logging.error(deb[0].tolist()[0])
        # exit(0)

        if global_step % log_freq == 0:
          logging.error(deb)
          # Copy before init to avoid overwritting.
          copy_model_parameters(sess, 'tkpg', 'tkpg_1')
          sess.run(val_it.initializer)
          sess.run(val_met_init_op_tensor)
          start_time = time()
          while True:
            try:
              sess.run(val_met_op_tensor)
            except tf.errors.OutOfRangeError:
              logging.error('validation Done.')
              break
            except KeyboardInterrupt:
              break
          logging.info("Step into %d...", global_step)
          # Get training metrics.
          acr, pd, pi_rec, pi_loss_met, beta_acc, beta_loss_met = sess.run([
              acr_tensor,
              pd_tensor,
              pi_rec_tensor,
              pi_loss_met_tensor,
              beta_acc_tensor,
              beta_loss_met_tensor,
          ])
          logging.info(
              "acr: %f, pd: omitted%s, "
              "pi loss: %f, beta loss: %f, "
              "raw pi loss: %f, avg raw pi loss: %f, "
              "avg beta loss: %f, pi rec: %f, beta acc: %f, "
              "importance_weight: (%f, %f, %f), "
              "batch time: %fs",
              acr, '', loss_pi, loss_beta,
              raw_loss_pi, pi_loss_met, beta_loss_met,
              pi_rec, beta_acc,
              np.min(iw), np.max(iw), np.mean(iw),
              batch_time)
          # Get validation metrics.
          acr, pd, pi_rec, pi_loss_met, beta_acc, beta_loss_met, val_beta_baseline_met, val_online_baseline_met = sess.run([
              val_acr_tensor,
              val_pd_tensor,
              val_pi_rec_tensor,
              val_pi_loss_met_tensor,
              val_beta_acc_tensor,
              val_beta_loss_met_tensor,
              val_beta_baseline_met_tensor,
              val_online_baseline_met_tensor,
          ])
          logging.info(
              "acr: %f, pd: omitted%s, "
              "pi loss: %f, beta loss: %f, "
              "raw pi loss: %f, avg raw pi loss: %f, "
              "avg beta loss: %f, pi rec: %f, beta acc: %f, "
              "beta baseline: %f, online baseline: %f, "
              "importance_weight: (%f, %f, %f), "
              "validation time: %fs",
              acr, '', loss_pi, loss_beta,
              raw_loss_pi, pi_loss_met, beta_loss_met,
              pi_rec, beta_acc,
              val_beta_baseline_met, val_online_baseline_met,
              np.min(iw), np.max(iw), np.mean(iw),
              time() - start_time)

          logging.info("Saving step %d...", global_step)
          saver.save(
              sess, os.path.join(basedir, "model"),
              global_step=global_step, write_meta_graph=True)
          # Reinitialize metrics.
          sess.run([met_init_op_tensor])

        deb = sess.run(tensors['deb'])
        if np.isnan(deb).any():
          logging.error(batch_idx)
          logging.error(deb.tolist())
          exit(1)
        # Actual training.
        start_time = time()
        _, global_step, loss_pi, loss_beta, raw_loss_pi, iw, _, deb = sess.run([
            train_op,
            global_step_tensor,
            loss_pi_tensor,
            loss_beta_tensor,
            raw_loss_pi_tensor,
            iw_tensor,
            met_op_tensor,
            tensors['deb'],
        ])
        batch_time = time() - start_time
        deb = sess.run(tensors['deb'])
        if np.isnan(deb).any():
          logging.error(batch_idx)
          logging.error(deb.tolist())
          exit(1)

        if max_train_steps is not None and global_step >= max_train_steps:
          break
      except tf.errors.OutOfRangeError:
        logging.error('Dataset exhausted.')
        break
      except KeyboardInterrupt:
        break
    logging.info("Saving step %d...", global_step)
    saver.save(
        sess,
        os.path.join(basedir, "model"),
        global_step=global_step,
        write_meta_graph=True)
    logging.info("Training done.")


def _run_testing(tensors, it_init_op, max_train_steps, log_freq=100):
  logging.info("Testing TKPG model...")
  training_basedir = _get_training_dir()

  user_id_tensor = tensors['user_id']
  # Inputs.
  actions_tensor = tensors['action_id']
  seq_mask_tensor = tensors['seq_mask']
  # Inference.
  prob_pi_tensor = tensors['prob_pi']
  # Metrics.
  met_op_tensor = tensors['met_op']
  met_init_op_tensor = tensors['met_init_op']
  acr_tensor = tensors['acr']
  pd_tensor = tensors['pd']
  pi_rec_tensor = tensors['pi_rec']
  pi_loss_met_tensor = tensors['pi_loss_met']
  beta_acc_tensor = tensors['beta_acc']
  beta_loss_met_tensor = tensors['beta_loss_met']
  pi_diff_action_tensor = tensors['pi_diff_action']

  with tfv1.Session() as sess:
    # Initialize all the variables.
    sess.run(tfv1.global_variables_initializer())
    sess.run(tfv1.tables_initializer())
    sess.run([met_init_op_tensor])
    # Restoring model.
    saver = tfv1.train.Saver(tfv1.trainable_variables())
    latest_path = tfv1.train.latest_checkpoint(training_basedir)
    saver.restore(sess, latest_path)
    for batch_idx in itertools.count():
      start_time = time()
      try:
        sess.run([
            met_op_tensor,
        ])
        # Get metrics. Varified that this will not fetch data.
        user_id, actions, seq_mask, prob_pi, acr, pd, pi_rec, pi_loss_met, beta_acc, beta_loss_met, pi_diff_action = sess.run([
            user_id_tensor,
            actions_tensor,
            seq_mask_tensor,
            prob_pi_tensor,
            acr_tensor,
            pd_tensor,
            pi_rec_tensor,
            pi_loss_met_tensor,
            beta_acc_tensor,
            beta_loss_met_tensor,
            pi_diff_action_tensor,
        ])
        if batch_idx % log_freq == 0:
          logging.info(
              "batch index: %d, acr: %f, pd: %s, "
              "avg raw pi loss: %f, avg beta loss: %f, "
              "pi rec: %f, beta acc: %f, batch time: %fs",
              batch_idx, acr, pd, pi_loss_met, beta_loss_met,
              pi_rec, beta_acc, time() - start_time)
          gather_pi_diff_action(
              user_id, pi_diff_action, actions, prob_pi,
              seq_mask)
      except tf.errors.OutOfRangeError:
        logging.error('Dataset exhausted.')
        break
      except KeyboardInterrupt:
        break
    logging.info("Testing done.")


def gather_pi_diff_action(user_id, pi_diff_action, actions, prob_pi, seq_mask):
  logging.error(user_id)
  logging.error(pi_diff_action)
  logging.error(actions)
  logging.error(np.argmax(prob_pi, axis=2))
  logging.error(seq_mask)
  logging.error(prob_pi)


def _export_for_serving(mc, serving_input, tensors):
  training_basedir = _get_training_dir()
  logging.info("Exporting TKPG model from %s", training_basedir)
  prob_pi = tensors['prob_pi']
  # prob_beta = tensors['prob_beta']
  h_state = tensors['h_state']
  all_action_ids = tensors['all_action_ids']
  deb = tensors['deb']
  with tfv1.Session() as sess:
    logging.error(tfv1.trainable_variables())
    saver = tfv1.train.Saver(tfv1.trainable_variables())
    sess.run(tfv1.global_variables_initializer())
    sess.run(tfv1.tables_initializer())

    latest_path = tfv1.train.latest_checkpoint(training_basedir)
    saver.restore(sess, latest_path)

    path = os.path.join(FLAGS.export_basedir, str(int(time())))
    logging.info("Export model to %s", path)
    builder = tfv1.saved_model.builder.SavedModelBuilder(path)
    builder.add_meta_graph_and_variables(
        sess,
        tags=[tfv1.saved_model.tag_constants.SERVING],
        signature_def_map={
            tfv1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tfv1.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    'example':
                    tfv1.saved_model.utils.build_tensor_info(serving_input)
                },
                outputs={
                    'pi_prob':
                    tfv1.saved_model.utils.build_tensor_info(prob_pi),
                    # 'beta_prob':
                    # tfv1.saved_model.utils.build_tensor_info(prob_beta),
                    'action_id':
                    tfv1.saved_model.utils.build_tensor_info(
                        tf.reshape(all_action_ids, [1, tf.shape(all_action_ids)[0]])),
                    'hidden_state':
                    tfv1.saved_model.utils.build_tensor_info(h_state),
                    'debug':
                    tfv1.saved_model.utils.build_tensor_info(deb),
                },
                method_name=tfv1.saved_model.signature_constants.
                PREDICT_METHOD_NAME,
            ),
        },
        main_op=tfv1.tables_initializer(),
        saver=saver)
    builder.save()


def main(_):
  mc_proto = mcpb.ModelConstants.FromString(
      open(FLAGS.model_constants_pb, "rb").read())
  mc = mc_lib.build_model_constants(mc_proto)
  if FLAGS.op == ops_lib.Op.TRAINING:
    tensors, val_tensors, val_it = _build_graph(
        FLAGS.op, mc, len(mc_proto.all_activity_key),
        files=sorted(glob.glob(FLAGS.trajectory_files)))
    _run_training(tensors, val_tensors, val_it, FLAGS.max_train_steps, log_freq=FLAGS.log_freq)
  elif FLAGS.op == ops_lib.Op.TESTING:
    tensors, init_op = _build_graph(
        FLAGS.op, mc, len(mc_proto.all_activity_key),
        files=sorted(glob.glob(FLAGS.trajectory_files)))
    _run_testing(tensors, init_op, FLAGS.max_train_steps, log_freq=FLAGS.log_freq)
  elif FLAGS.op == ops_lib.Op.EXPORT_SERVING:
    serving_input, tensors = _build_graph(
        FLAGS.op, mc, len(mc_proto.all_activity_key))
    _export_for_serving(mc, serving_input, tensors)
  else:
    logging.fatal("Unsupported --op=%s", FLAGS.op)


if __name__ == "__main__":
  app.run(main)
