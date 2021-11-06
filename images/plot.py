from __future__ import absolute_import, division, print_function

from absl import app, flags, logging

import gym
import time
import queue
import random
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from collections import deque, defaultdict
import matplotlib.pyplot as plt

tfv1.disable_eager_execution()

import client.client as client_lib
import env.halfcheetahv2 as halfcheetahv2_lib
import model.rl.agent.agent as agent_lib
import model.rl.agent.critic as critic_lib
import model.rl.agent.trpo as trpo_lib

FLAGS = flags.FLAGS

flags.DEFINE_string("op", "Train", "Train or Test?")
flags.DEFINE_integer("batch_size", 32, "Sample size for one batch.")
flags.DEFINE_integer("num_epoches", 1, "Maximum number of training epoches.")
flags.DEFINE_integer("max_train_steps", None,
                     "If not None, run at most this many steps.")
flags.DEFINE_integer("max_seq_len", 256, "Maximum sequence length.")
flags.DEFINE_integer("init_action", -1, "The initial action taken.")
flags.DEFINE_integer("num_clients", 10, "The number of clients.")


def main(_):
  # Initialize optimizer.
  num_client = 1

  # HalfCheetah-V2
  num_iter = 500
  timestep_per_batch = 2048
  num_epoch = 10
  batch_size = 64
  lr = 1e-4

  # Create env before hand for saving memory.
  envs = []
  for i in range(num_client):
    seed = int(i * 1e4)
    env = halfcheetahv2_lib.HalfCheetahV2(seed, parallel=12)
    envs.append(env)

  # Create agent.
  seed = int(i * 1e4)
  optimizer = tf.optimizers.Adam(learning_rate=lr)
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
                              dropout_rate=0.05)
  agent = agent_lib.Agent(
      str(i), policy, init_exp=0.1, final_exp=0.0, anneal_steps=1,
      critic=critic_lib.Critic(env.state_dim, 10))
  client = client_lib.Client(i, i, agent, env)

  start = time.time()
  paths, episode_rewards = client.rollout(
      1e6, client.env.get_parallel_envs())
  xs = defaultdict(list)
  for trajectory in paths:
    for obs in trajectory['observations']:
      for i in range(len(obs)):
        xs[i].append(obs[i])
  for i in range(17):
    plt.clf()
    plt.hist(xs[i], bins=10000)
    plt.savefig('./images/obs_dim_%d.png' % (i))
  print(time.time() - start)


if __name__ == "__main__":
  app.run(main)
