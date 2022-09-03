from __future__ import absolute_import, division, print_function

from absl import app, flags, logging

import gym
import time
import queue
import random
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from collections import deque
from multiprocessing.dummy import Pool as ThreadPool

tfv1.disable_eager_execution()

import client.client as client_lib
import environment.airraidramv0 as airraidramv0_lib
import environment.antv2 as antv2_lib
import environment.cartpolev0 as cartpolev0_lib
import environment.fetchpickandplacev1 as fetchpickandplacev1_lib
import environment.halfcheetahv2 as halfcheetahv2_lib
import environment.hopperv2 as hopperv2_lib
import environment.humanoidv2 as humanoidv2_lib
import environment.invertedpendulumv2 as invertedpendulumv2_lib
import environment.reacherv2 as reacherv2_lib
import environment.walker2dv2 as walker2dv2_lib
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
flags.DEFINE_integer("max_train_steps", None,
                     "If not None, run at most this many steps.")
flags.DEFINE_integer("max_seq_len", 256, "Maximum sequence length.")
flags.DEFINE_integer("init_action", -1, "The initial action taken.")
flags.DEFINE_integer("num_clients", 10, "The number of clients.")


def experiment(client):
  MAX_EPISODES = 1000
  MAX_STEPS    = 200
  MAX_EPISODES = 100
  MAX_STEPS    = 10000
  client.experiment(MAX_EPISODES, MAX_STEPS)


def main(_):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  logging.error(gpus)

  # Initialize optimizer.
  num_client = 1

  num_iter = 300
  timestep_per_batch = 1
  num_epoch = 1
  batch_size = 1

  num_iter = 200
  timestep_per_batch = 5000
  num_epoch = 10
  batch_size = 128

  # Humanoid-V2
  num_iter = 5000
  timestep_per_batch = 10000
  num_epoch = 15
  batch_size = 4096
  lr = 1e-4

  # HalfCheetah-V2
  num_iter = 300
  timestep_per_batch = 2048
  num_epoch = 10
  batch_size = 64
  lr = 1e-3
  lr = 5e-4
  lr = 3e-4
  lr = 1e-4
  lr = 1e-2

  lr = 3e-3
  kl_targ = 1e-2  # 17.26.
  kl_targ = 1e-3  # 48.75.
  kl_targ = 1e-1  # 
  lr = 1e-3
  kl_targ = 1e-1  # 22.53.
  kl_targ = 1e-3  # 23.83.
  kl_targ = 1e-2  # 10.56. 11.49;12.34;10.77.
  # Create env before hand for saving memory.
  num_client = 64
  num_client = 5
  envs = []
  for i in range(num_client):
    seed = int(i * 1e4)
    # env = cartpolev0_lib.CartPoleV0(seed)
    # env = airraidramv0_lib.AirRaidRamV0(seed)
    # env = fetchpickandplacev1_lib.FetchPickAndPlaceV1(seed)
    # env = invertedpendulumv2_lib.InvertedPendulumV2(seed)
    # env = antv2_lib.AntV2(seed)
    # env = hopperv2_lib.HopperV2(seed)
    # env = humanoidv2_lib.HumanoidV2(seed)
    # env = walker2dv2_lib.Walker2dV2(seed)
    # env = halfcheetahv2_lib.HalfCheetahV2(seed)
    # env = reacherv2_lib.ReacherV2(seed, qpos_high_low=[[0.10000000000000003, 0.15000000000000002], [0.10, 0.15]])
    # env = reacherv2_lib.ReacherV2(seed)
    qpos, noise = reacherv2_lib.generate_reacher_heterogeneity(i, 'both')
    qpos, noise = reacherv2_lib.generate_reacher_heterogeneity(i, 'iid')
    qpos, noise = reacherv2_lib.generate_reacher_heterogeneity(i, 'dynamics')
    # qpos, _ = reacherv2_lib.generate_reacher_heterogeneity(0, 'init-state')
    # noise = np.zeros(shape=(2,))
    # qpos, noise = reacherv2_lib.generate_reacher_heterogeneity(i, 'iid')
    env = reacherv2_lib.ReacherV2(
        seed=0, qpos_high_low=qpos, qvel_high_low=[-0.005, 0.005],
        action_noise=noise)
    logging.error(qpos)
    logging.error(noise)

    envs.append(env)

  fl_params = {
      'clients_per_round': num_client,
      'num_rounds': 1,
      'sigma': 0,
      # The more local iteration, the more likely for FedAvg to diverge.
      'num_iter': 0,
      'timestep_per_batch': 0,
      'max_steps': 10000,
      'eval_every': 0,
      'drop_percent': 0.0,
      'has_global_svf': True,
      'verbose': True,
      'svf_n_timestep': 1e4,
      # Tuned for Reacher-V2. Optional.
      'retry_min': 0,
      # CSV for saving reward_history.
      'reward_history_fn': '',
  }
  fl = fedtrpo_lib.FedTRPO(**fl_params)

  # Create agent.
  clients = []
  d_as = []
  for i in range(num_client):
    seed = int(i * 1e4)

    optimizer = pgd_lib.PerturbedGradientDescent(lr, mu=1e-5)
    optimizer = tf.optimizers.SGD(learning_rate=lr)
    optimizer = tf.optimizers.Adam(learning_rate=3e-4)

    env = envs[i]

    policy = trpo_lib.TRPOActor(env,
                                optimizer,
                                model_scope='trpo_' + str(i),
                                batch_size=batch_size,
                                num_epoch=num_epoch,
                                future_discount=0.99,
                                kl_targ=kl_targ,
                                beta=1.0,
                                lam=0.95,
                                importance_weight_cap=100,
                                dropout_rate=0.05,
                                distance_metric='sqrt_kl',
                                linear=False,
                                verbose=False)
    agent = agent_lib.Agent(
        str(i), policy, init_exp=0.1, final_exp=0.0, anneal_steps=1,
        critic=critic_lib.Critic(env.state_dim, 10), expose_critic=True)
    # policy = reinforce_lib.REINFORCEActor(env,
    #                                       optimizer,
    #                                       model_scope='reinforce_' + str(i),
    #                                       batch_size=batch_size,
    #                                       future_discount=0.99)
    # agent = agent_lib.Agent(str(i), policy, init_exp=0.1, final_exp=0.0,
    #                         anneal_steps=500)
    client = client_lib.Client(
        i, i, agent, env, num_test_epochs=20, parallel=10, filt=True,
        extra_features=['probs'])
    # client.enable_svf(2e3)
    # d_a = client.get_da()
    # d_as.append(d_a)
    # logging.error(len(d_a))
    clients.append(client)
    fl.register(client)

  # norm_bs, norm_das, norm_avgs = fl.get_heterogeneity_level(d_as, logger=print)
  # print(norm_bs)
  # print(norm_das)
  # print(norm_avgs)
  # exit(0)

  pool = ThreadPool(num_client)
  # results = pool.map(experiment, clients)
  res = []
  for client in clients:
    for i in range(num_iter):
      if i % 10 == 0:
        logging.error('%s: %s' % (i, client.test()))
      client.experiment(
          1, timestep_per_batch,
          callback_before_fit=[client.sync_old_policy],
          logger=logging.error)
    res.append(client.test())
  logging.error(np.mean(res))


if __name__ == "__main__":
  app.run(main)
