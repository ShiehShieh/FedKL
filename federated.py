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
  num_iter = 500
  timestep_per_batch = 2048
  num_epoch = 10
  batch_size = 64
  lr = 1e-3
  lr = 5e-4
  lr = 3e-4
  lr = 1e-4
  lr = 1e-3

  # Create env before hand for saving memory.
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
    env = reacherv2_lib.ReacherV2(seed, qpos_high_low=[[0.10000000000000003, 0.15000000000000002], [0.10, 0.15]])
    # env = reacherv2_lib.ReacherV2(seed)
    envs.append(env)

  # Create agent.
  clients = []
  for i in range(num_client):
    seed = int(i * 1e4)

    optimizer = pgd_lib.PerturbedGradientDescent(lr, mu=1e-5)
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    optimizer = tf.optimizers.SGD(learning_rate=lr)

    env = envs[i]

    policy = trpo_lib.TRPOActor(env,
                                optimizer,
                                model_scope='trpo_' + str(i),
                                batch_size=batch_size,
                                num_epoch=num_epoch,
                                future_discount=0.99,
                                kl_targ=0.01,
                                beta=1.0,
                                lam=0.95,
                                importance_weight_cap=100,
                                dropout_rate=0.05,
                                linear=False,
                                verbose=False)
    agent = agent_lib.Agent(
        str(i), policy, init_exp=0.1, final_exp=0.0, anneal_steps=1,
        critic=critic_lib.Critic(env.state_dim, 10))
    # policy = reinforce_lib.REINFORCEActor(env,
    #                                       optimizer,
    #                                       model_scope='reinforce_' + str(i),
    #                                       batch_size=batch_size,
    #                                       future_discount=0.99)
    # agent = agent_lib.Agent(str(i), policy, init_exp=0.1, final_exp=0.0,
    #                         anneal_steps=500)
    client = client_lib.Client(
        i, i, agent, env, num_test_epochs=20, parallel=10, filt=True,
        extra_features=[])
    clients.append(client)

  pool = ThreadPool(num_client)
  # results = pool.map(experiment, clients)
  res = []
  for client in clients:
    for i in range(num_iter):
      if i % 10 == 0:
        logging.error(client.test())
      client.experiment(
          1, timestep_per_batch,
          callback_before_fit=[client.sync_old_policy],
          logger=logging.error)
    res.append(client.test())
  logging.error(np.mean(res))


if __name__ == "__main__":
  app.run(main)
