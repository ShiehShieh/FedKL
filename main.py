from __future__ import absolute_import, division, print_function

from absl import app, flags, logging

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

tfv1.disable_eager_execution()

import client.client as client_lib
import environment.halfcheetahv2 as halfcheetahv2_lib
import environment.reacherv2 as reacherv2_lib
import model.fl.fedavg as fedavg_lib
import model.fl.fedprox as fedprox_lib
import model.fl.fedtrpo as fedtrpo_lib
import model.rl.agent.agent as agent_lib
import model.rl.agent.critic as critic_lib
import model.rl.agent.reinforce as reinforce_lib
import model.rl.agent.trpo as trpo_lib
import model.optimizer.pgd as pgd_lib

FLAGS = flags.FLAGS

flags.DEFINE_string("op", "Train", "Train or Test?")
flags.DEFINE_integer("batch_size", 32, "Sample size for one batch.")
flags.DEFINE_integer("num_epoches", 1, "Maximum number of training epoches.")
flags.DEFINE_integer("clients_per_round", 5, "The number of clients.")
flags.DEFINE_integer("num_rounds", 500, "The number of FL rounds.")
flags.DEFINE_integer("n_local_iter", 200, "The number of local updates per round.")
flags.DEFINE_string("heterogeneity_type", "init-state", "iid, init-state, dynamics or both?")
flags.DEFINE_bool("expose_critic", False, "If true, critic will be federated, too.")
flags.DEFINE_integer("eval_every", 1, "Perform a test run every this round.")

flags.DEFINE_float("lr", 1e-3, "Learning rate.")
flags.DEFINE_float("mu", 1e-3, "Penalty coefficient for FedProx.")
flags.DEFINE_float("nm_targ", 1e-3, "norm penalty target of FedTRPO.")
flags.DEFINE_bool("has_global_svf", False, "If true, client has access to the global state visitation frequency.")
flags.DEFINE_string("distance_metric", 'tv', "One of tv, mahalanobis and wasserstein.")
flags.DEFINE_string("fed", "FedAvg", "Federated Learning Algorithm.")
flags.DEFINE_string("pg", "REINFORCE", "Policy Gradient Algorithm.")
flags.DEFINE_string("env", "halfcheetah", "halfcheetah, reacher or figureeightv1.")

flags.DEFINE_bool("linear", False, "Use linear layer for MLP.")
flags.DEFINE_integer("parallel", 10, "Parallelism for env rollout.")
flags.DEFINE_float("svf_n_timestep", 1e6, "The number of timestep for estimating state visitation frequency.")
flags.DEFINE_string("reward_history_fn", "", "The file stored reward history.")

flags.DEFINE_float("retry_min", -30, "local objective exceeded this cost will be considered as diverged.")

np.random.seed(0)
tf.random.set_seed(0)


def generate_halfcheetah_heterogeneity(i):
  x_left, x_right = -0.005, 0.005
  gravity = -9.81
  if FLAGS.heterogeneity_type == 'iid':
    pass
  if FLAGS.heterogeneity_type == 'init-state':
    # 50 clients, and wider range of each initial state.
    interval = 0.02
    x_left = -0.5 + interval * 1.0 * i
    # 30 clients, and standard range of each initial state.
    interval = 0.01
    x_left = -0.5 + interval * (10.0 / 3.0) * i
    #
    x_right = x_left + interval
  if FLAGS.heterogeneity_type == 'dynamics':
    low = -20
    (i + 1) / num_total_clients
    gravity = float(i + 1) / float(num_total_clients) * low
  return x_left, x_right, gravity

  # Worth to try.
  interval = 0.01
  x_left = -0.2 + interval * (4.0 / 3.0) * i

  interval = 0.01
  x_left = -0.1 + interval * 1 * i

  # 100 clients.
  interval = 0.01
  x_left = -0.375 + interval * 3.0 / 4.0 * i
  x_right = x_left + interval
  return x_left, x_right


def generate_reacher_heterogeneity(i):
  out = [[-0.2, 0.2], [-0.2, 0.2]]
  action_noise = np.zeros(shape=(2,))
  if FLAGS.heterogeneity_type == 'iid':
    pass
  if FLAGS.heterogeneity_type in ['init-state', 'both']:
    # 64 clients.
    if i > 63:
      raise NotImplementedError
    j = i
    if i in [0, 7, 56, 63]:
      j = 1
    row = j // 8
    col = j % 8
    x = -0.2 + row * 0.05
    y = 0.2 - col * 0.05
    out = [[x, x + 0.05], [y - 0.05, y]]
    #
  if FLAGS.heterogeneity_type in ['dynamics', 'both']:
    action_noise = np.random.normal(0.002, 0.0005, 2)
  if out[0][0] > out[0][1] or out[1][0] > out[1][1]:
    raise NotImplementedError
  return out, action_noise


def main(_):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  logging.error(gpus)

  # Create env before hand for saving memory.
  envs = []
  # Keep this number low or we may fail to simulate the heterogeneity.
  num_total_clients = 64
  universial_client = None
  if FLAGS.env == 'figureeightv1':
    num_total_clients = 7
  for i in range(num_total_clients):
    seed = int(i * 1e4)
    if FLAGS.env == 'halfcheetah':
      x_left, x_right, gravity = generate_halfcheetah_heterogeneity(i)
      env = halfcheetahv2_lib.HalfCheetahV2(
          seed=seed, qpos_high_low=[x_left, x_right],
          qvel_high_low=[-0.005, 0.005], gravity=gravity)
      logging.error([x_left, x_right])
    if FLAGS.env == 'reacher':
      # Numpy is already seeded.
      qpos, noise = generate_reacher_heterogeneity(i)
      env = reacherv2_lib.ReacherV2(
          seed=seed, qpos_high_low=qpos, qvel_high_low=[-0.005, 0.005],
          action_noise=noise)
      logging.error(qpos)
      logging.error(noise)
    if FLAGS.env == 'figureeightv1':
      import logging as py_logging
      py_logging.disable(py_logging.INFO)
      import environment.figureeight as figureeight_lib
      env = figureeight_lib.CustomizedCAV()
      if universial_client is None:
        fev1 = figureeight_lib.FlowFigureEightV1(0)
        universial_client = client_lib.UniversalClient(
            envs=fev1, future_discount=0.99, lam=0.95, num_test_epochs=20
        )

    envs.append(env)

  # Federated Learning Experiments.
  lr = FLAGS.lr
  mu = FLAGS.mu
  fl_params = {
      'clients_per_round': FLAGS.clients_per_round,
      'num_rounds': FLAGS.num_rounds,
      # The more local iteration, the more likely for FedAvg to diverge.
      'num_iter': FLAGS.n_local_iter,
      'timestep_per_batch': 1500,  # 2048,
      'max_steps': 10000,
      'eval_every': FLAGS.eval_every,
      'drop_percent': 0.0,
      'has_global_svf': FLAGS.has_global_svf,
      'verbose': True,
      'svf_n_timestep': FLAGS.svf_n_timestep,
      # Tuned for Reacher-V2. Optional.
      'retry_min': FLAGS.retry_min,
      # CSV for saving reward_history.
      'reward_history_fn': FLAGS.reward_history_fn,
  }
  sigma = 0.0
  if FLAGS.fed == 'FedAvg':
    fl = fedavg_lib.FedAvg(**fl_params)
    # opt_class = lambda: tf.optimizers.Adam(learning_rate=lr)
    opt_class = lambda: tf.optimizers.SGD(learning_rate=lr)
  elif FLAGS.fed == 'FedProx':
    fl = fedprox_lib.FedProx(**fl_params)
    opt_class = lambda: pgd_lib.PerturbedGradientDescent(
        learning_rate=lr, mu=mu)
  elif FLAGS.fed == 'FedTRPO':
    fl = fedtrpo_lib.FedTRPO(**fl_params)
    opt_class = lambda: tf.optimizers.SGD(learning_rate=lr)
    sigma = 1.0
  fl.register_universal_client(universial_client)

  # Set up clients.
  for i in range(num_total_clients):
    seed = int(i * 1e4)
    env = envs[i]
    optimizer = opt_class()
    if FLAGS.pg == 'REINFORCE':
      agent = agent_lib.Agent(
          str(i), reinforce_lib.REINFORCEActor(
              env, optimizer, model_scope='reinforce' + str(i),
              batch_size=1, future_discount=0.99,
          ), init_exp=0.5, final_exp=0.0, anneal_steps=500, 
      )
    elif FLAGS.pg == 'TRPO':
      # Seeding in order to avoid randomness.
      agent = agent_lib.Agent(
          str(i), trpo_lib.TRPOActor(
              env, optimizer, model_scope='trpo_' + str(i), batch_size=64,
              num_epoch=10, future_discount=0.99, kl_targ=0.01, beta=1.0,
              lam=0.95, seed=seed, linear=FLAGS.linear, verbose=False,
              nm_targ=FLAGS.nm_targ, sigma=sigma,
              distance_metric=FLAGS.distance_metric,
          ), init_exp=0.5, final_exp=0.0, anneal_steps=1,
          critic=critic_lib.Critic(env.state_dim, 200, seed=seed),
          expose_critic=FLAGS.expose_critic
      )

    client = client_lib.Client(
        i, 0, agent, env, num_test_epochs=20, parallel=FLAGS.parallel,
        filt=True, extra_features=set([]))
    fl.register(client)

  # Start FL training.
  reward_history = fl.train()

  # Logging.
  logging.error('# retry: %d' % (fl.get_num_retry()))

  # Cleanup.
  figureeight_lib.cleanup()
  if universial_client is not None:
    universial_client.cleanup()


if __name__ == "__main__":
  app.run(main)
