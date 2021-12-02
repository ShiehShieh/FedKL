from __future__ import absolute_import, division, print_function

from absl import app, flags, logging

from collections import defaultdict

import time
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

tfv1.disable_eager_execution()

import client.client as client_lib
import environment.halfcheetahv2 as halfcheetahv2_lib
import environment.reacherv2 as reacherv2_lib
import model.rl.agent.agent as agent_lib
import model.rl.agent.critic as critic_lib
import model.rl.agent.trpo as trpo_lib
import model.fl.fedtrpo as fedtrpo_lib
import model.rl.comp.state_visitation_frequency as svf_lib

FLAGS = flags.FLAGS

flags.DEFINE_float("n_steps", "1e6", "The number of rollout timesteps.")


def main(_):
  lr = 1e-4
  parallel = 8

  # Create env before hand for saving memory.
  envs = []
  num_total_clients = 64
  num_total_clients = 16
  for i in range(num_total_clients):
    seed = int(i * 1e4)
    interval = 0.01
    # Little heterogeneity.
    x_left = -0.2 + interval * (4.0 / 3.0) * 3 * i
    # Severe heterogeneity.
    interval = 0.02
    x_left = -0.5 + interval * 1.0 * 5.0 * i
    x_right = x_left + interval
    # Severe heterogeneity.
    interval = 0.01
    x_left = -0.5 + interval * (10.0 / 3.0) * 3.0 * i
    x_right = x_left + interval
    env = halfcheetahv2_lib.HalfCheetahV2(
        seed=seed, qpos_high_low=[x_left, x_right],
        qvel_high_low=[-0.005, 0.005], gravity=-9.81)
    #
    j = i
    if i in [0, 7, 56, 63]:
      j = 1

    row = j // 8
    col = j % 8
    x = -0.2 + row * 0.05
    y = 0.2 - col * 0.05
    logging.error([i, row, col, [[x, x + 0.05], [y, y - 0.05]]])
    env = reacherv2_lib.ReacherV2(
        seed=seed, qpos_high_low=[[x, x + 0.05], [y, y - 0.05]],
        qvel_high_low=[-0.005, 0.005])
    # for j in range(1000):
    #   env.render()
    #   time.sleep(0.1)
    #   obs, rew, done, info = env.step(env.env.action_space.sample())
    #   print(j, obs, rew, done, info)
    # exit(0)
    # logging.error([x_left, x_right])
    envs.append(env)

  # Not going to train it anyway.
  optimizer = tf.optimizers.Adam(learning_rate=lr)
  agent = agent_lib.Agent(
      '0', trpo_lib.TRPOActor(
          env, optimizer, model_scope='trpo_' + str(i),
          batch_size=64, num_epoch=10, future_discount=0.99,
          kl_targ=0.01, beta=1.0, lam=0.95, seed=0,
          verbose=False, linear=True,
      ), init_exp=0.5, final_exp=0.0, anneal_steps=1,
      critic=critic_lib.Critic(env.state_dim, 200, seed=0)
  )
  params = {
      'clients_per_round': 10,
      'num_rounds': 200,
      'num_iter': 10,
      'timestep_per_batch': 2048,
      'max_steps': 10000,
      'eval_every': 1,
      'drop_percent': 0.0,
      'verbose': True,
      'svf_n_timestep': FLAGS.n_steps,
  }
  fl = fedtrpo_lib.FedTRPO(**params)

  # Set up clients.
  svfs = []
  svf_ms = []
  clients = []
  for i in range(num_total_clients):
    seed = int(i * 1e4)
    env = envs[i]
    client = client_lib.Client(
        i, 0, agent, env, num_test_epochs=2, filt=True, parallel=parallel,
        extra_features=[])
    clients.append(client)
    fl.register(client)
  svfs, _ = fl.get_state_visitation_frequency(clients, logging.error)
  np.save('./svfs.npy', svfs)

  from sklearn.metrics.pairwise import cosine_similarity
  for i in cosine_similarity(svfs, svfs):
    print(['%.5f' % f for f in i])


if __name__ == "__main__":
  app.run(main)
