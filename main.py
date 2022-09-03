from __future__ import absolute_import, division, print_function

from absl import app, flags, logging

import random
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

tfv1.disable_eager_execution()

import client.client as client_lib
import environment.halfcheetahv2 as halfcheetahv2_lib
import environment.mountaincarcontinuous as mountaincarcontinuous_lib
import environment.reacherv2 as reacherv2_lib
import model.fl.fedavg as fedavg_lib
import model.fl.fedprox as fedprox_lib
import model.fl.fedtrpo as fedtrpo_lib
import model.fl.fmarl as fmarl_lib
import model.rl.agent.agent as agent_lib
import model.rl.agent.critic as critic_lib
import model.rl.agent.reinforce as reinforce_lib
import model.rl.agent.trpo as trpo_lib
import model.optimizer.dbpg as dbpg_lib
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

flags.DEFINE_integer("init_seed", 0, "Seed for parameter initialization.")
flags.DEFINE_float("lr", 1e-3, "Learning rate.")
flags.DEFINE_float("mu", 1e-3, "Penalty coefficient for FedProx.")
flags.DEFINE_float("sigma", 1e0, "Penalty coefficient for FedPG.")
flags.DEFINE_bool("fixed_sigma", False, "If true, fixed sigma, else adaptive sigma.")
flags.DEFINE_float("kl_targ", 1e-2, "KL divergence target of FedTRPO.")
flags.DEFINE_float("nm_targ", 1e-3, "Norm penalty target of FedTRPO.")
flags.DEFINE_float("lambda_dc", 0.98, "Decay constant of FMARL.")
flags.DEFINE_bool("disable_kl", False, "Turn off kl penalty.")
flags.DEFINE_bool("disable_tv", False, "Turn off tv penalty.")
flags.DEFINE_bool("has_global_svf", False, "If true, client has access to the global state visitation frequency.")
flags.DEFINE_string("distance_metric", 'tv', "One of tv, sqrt_kl, mahalanobis and wasserstein.")
flags.DEFINE_string("fed", "FedAvg", "Federated Learning Algorithm.")
flags.DEFINE_string("pg", "REINFORCE", "Policy Gradient Algorithm.")
flags.DEFINE_string("env", "halfcheetah", "halfcheetah, reacher, mcc or figureeightv1.")

flags.DEFINE_bool("is_centralized", False, "If true, use centralized training.")
flags.DEFINE_bool("linear", False, "Use linear layer for MLP.")
flags.DEFINE_integer("parallel", 10, "Parallelism for env rollout.")
flags.DEFINE_float("svf_n_timestep", 1e6, "The number of timestep for estimating state visitation frequency.")
flags.DEFINE_bool("eval_heterogeneity", False, "If true, evaluate the level of heterogeneity.")
flags.DEFINE_string("reward_history_fn", "", "The file stored reward history.")
flags.DEFINE_string("b_history_fn", "", "The file stored B matrix norm history.")
flags.DEFINE_string("da_history_fn", "", "The file stored DxA matrix norm history.")
flags.DEFINE_string("avg_history_fn", "", "The file stored \sum{DxA} matrix norm history.")

flags.DEFINE_float("retry_min", -30, "local objective exceeded this cost will be considered as diverged.")

random.seed(0)
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


def main(_):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  logging.error(gpus)

  # Create env before hand for saving memory.
  envs = []
  # Keep this number low or we may fail to simulate the heterogeneity.
  num_total_clients = 64
  universial_client = None
  timestep_per_batch = 2048
  if FLAGS.env == 'figureeightv1':
    num_total_clients = 7
  if FLAGS.env == 'figureeightv2':
    num_total_clients = 14
  if FLAGS.env == 'mcc':
    num_total_clients = 10
    num_total_clients = 1
  for i in range(num_total_clients):
    # if FLAGS.env == 'reacher':
    #   if i % 8 > 3 or i // 8 > 3:
    #     continue
    seed = int(i * 1e4)
    if FLAGS.env == 'halfcheetah':
      x_left, x_right, gravity = generate_halfcheetah_heterogeneity(i)
      env = halfcheetahv2_lib.HalfCheetahV2(
          seed=seed, qpos_high_low=[x_left, x_right],
          qvel_high_low=[-0.005, 0.005], gravity=gravity)
      logging.error([x_left, x_right])
    if FLAGS.env == 'reacher':
      # Numpy is already seeded.
      qpos, noise = reacherv2_lib.generate_reacher_heterogeneity(
          i, FLAGS.heterogeneity_type)
      env = reacherv2_lib.ReacherV2(
          seed=seed, qpos_high_low=qpos, qvel_high_low=[-0.005, 0.005],
          action_noise=noise)
      logging.error(qpos)
      logging.error(noise)
    if FLAGS.env == 'mcc':
      # Numpy is already seeded.
      qpos, noise = mountaincarcontinuous_lib.generate_mountaincarcontinuous_heterogeneity(
          i, FLAGS.heterogeneity_type)
      env = mountaincarcontinuous_lib.MountianCarContinuous(
          seed=seed, qpos_low_high=qpos, action_noise=noise)
      logging.error(qpos)
      logging.error(noise)
    if FLAGS.env.startswith('figureeight'):
      timestep_per_batch = 1500 * 1
      import logging as py_logging
      py_logging.disable(py_logging.INFO)
      import environment.figureeight as figureeight_lib
      env = figureeight_lib.CustomizedCAV()
      if universial_client is None:
        fev = None
        if FLAGS.env == 'figureeightv1':
          fev = figureeight_lib.FlowFigureEightV1(0)
        elif FLAGS.env == 'figureeightv2':
          fev = figureeight_lib.FlowFigureEightV2(0)
        else:
          raise NotImplementedError
        # TODO(XIE,Zhijie): Set num_test_epochs to 40 for report.
        universial_client = client_lib.UniversalClient(
            envs=fev, future_discount=0.99, lam=0.95, num_test_epochs=40
        )

    envs.append(env)
  num_total_clients = len(envs)

  # Federated Learning Experiments.
  lr = FLAGS.lr
  fl_params = {
      'clients_per_round': FLAGS.clients_per_round,
      'num_rounds': FLAGS.num_rounds,
      'sigma': FLAGS.sigma,
      # The more local iteration, the more likely for FedAvg to diverge.
      'num_iter': FLAGS.n_local_iter,
      'timestep_per_batch': timestep_per_batch,
      'max_steps': 10000,
      'eval_every': FLAGS.eval_every,
      'drop_percent': 0.0,
      'has_global_svf': FLAGS.has_global_svf,
      'verbose': True,
      'svf_n_timestep': FLAGS.svf_n_timestep,
      'eval_heterogeneity': FLAGS.eval_heterogeneity,
      # Tuned for Reacher-V2. Optional.
      'retry_min': FLAGS.retry_min,
      # CSV for saving reward_history.
      'reward_history_fn': FLAGS.reward_history_fn,
      # CSV for saving B matrix norm.
      'b_history_fn': FLAGS.b_history_fn,
      # CSV for saving DxA matrix norm.
      'da_history_fn': FLAGS.da_history_fn,
      # CSV for saving \sum{DxA} matrix norm.
      'avg_history_fn': FLAGS.avg_history_fn,
  }
  beta = 1.0
  sigma = 0.0
  mu = 0.0
  if FLAGS.fed == 'FedAvg':
    fl = fedavg_lib.FedAvg(**fl_params)
    # opt_class = lambda: tf.optimizers.Adam(learning_rate=lr)
    opt_class = lambda: tf.optimizers.SGD(learning_rate=lr)
  elif FLAGS.fed == 'FedProx':
    fl = fedprox_lib.FedProx(**fl_params)
    opt_class = lambda: pgd_lib.PerturbedGradientDescent(
        learning_rate=lr, mu=FLAGS.mu)
    opt_class = lambda: tf.optimizers.SGD(learning_rate=lr)
    mu = FLAGS.mu
  elif FLAGS.fed == 'FedTRPO':
    fl = fedtrpo_lib.FedTRPO(**fl_params)
    opt_class = lambda: tf.optimizers.SGD(learning_rate=lr)
    sigma = FLAGS.sigma
  elif FLAGS.fed == 'FMARL':
    fl = fmarl_lib.FMARL(**fl_params)
    opt_class = lambda: dbpg_lib.DecayBasedGradientDescent(
        learning_rate=lr, lamb=FLAGS.lambda_dc)
  fl.register_universal_client(universial_client)
  if FLAGS.disable_kl:
    beta = 0.0
  if FLAGS.disable_tv:
    sigma = 0.0

  # Set up clients.
  for i in range(num_total_clients):
    # Use the same seed for all agents.
    seed = FLAGS.init_seed
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
      gradient_clip_norm = 0.5  # Reacher-v2.
      gradient_clip_norm = 100.0  # Reacher-v2.
      gradient_clip_norm = None  # Reacher-v2.
      agent = agent_lib.Agent(
          str(i), trpo_lib.TRPOActor(
              env, optimizer, model_scope='trpo_' + str(i), batch_size=64,
              num_epoch=10, future_discount=0.99, kl_targ=FLAGS.kl_targ,
              beta=beta, lam=0.95, seed=seed, linear=FLAGS.linear,
              verbose=False, nm_targ=FLAGS.nm_targ, sigma=sigma,
              # nm_targ_adap=(1.0, 0.1, 20),
              nm_targ_adap=(FLAGS.nm_targ, FLAGS.nm_targ, 50),
              fixed_sigma=FLAGS.fixed_sigma, mu=mu,
              gradient_clip_norm=gradient_clip_norm,
              distance_metric=FLAGS.distance_metric,
          ), init_exp=0.5, final_exp=0.0, anneal_steps=1,
          critic=critic_lib.Critic(env.state_dim, 200, seed=seed),
          expose_critic=FLAGS.expose_critic
      )

    if FLAGS.is_centralized and fl.num_clients() > 0:
      fl.register(fl.get_client(0))
      continue

    client = client_lib.Client(
        i, 0, agent, env, num_test_epochs=20, parallel=FLAGS.parallel,
        filt=True, extra_features=set([]))
    if FLAGS.eval_heterogeneity:
      client.enable_svf(FLAGS.svf_n_timestep)
    fl.register(client)

  # Start FL training.
  reward_history = fl.train()

  # Logging.
  logging.error('# retry: %d' % (fl.get_num_retry()))

  # Cleanup.
  if universial_client is not None:
    figureeight_lib.cleanup()
    universial_client.cleanup()


if __name__ == "__main__":
  app.run(main)
