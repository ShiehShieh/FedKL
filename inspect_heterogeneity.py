from __future__ import absolute_import, division, print_function

from absl import app, flags, logging

from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

tfv1.disable_eager_execution()

import client.client as client_lib
import env.halfcheetahv2 as halfcheetahv2_lib
import model.rl.agent.agent as agent_lib
import model.rl.agent.critic as critic_lib
import model.rl.agent.trpo as trpo_lib
import model.rl.comp.state_visitation_frequency as svf_lib

FLAGS = flags.FLAGS

flags.DEFINE_float("lr", "1e-4", "Learning rate.")
flags.DEFINE_string("fed", "FedAvg", "Federated Learning Algorithm.")
flags.DEFINE_string("pg", "REINFORCE", "Policy Gradient Algorithm.")


def main(_):
  lr = FLAGS.lr
  parallel = 8

  # Create env before hand for saving memory.
  envs = []
  num_total_clients = 10
  for i in range(num_total_clients):
    seed = int(i * 1e4)
    interval = 0.01
    x_left = -0.0375 + interval * 3.0 / 4.0 * i
    x_right = x_left + interval
    env = halfcheetahv2_lib.HalfCheetahV2(
        seed=seed, qpos_high_low=[x_left, x_right],
        qvel_high_low=[-0.005, 0.005], parallel=parallel)
    logging.error([x_left, x_right])
    envs.append(env)

  # Not going to train it anyway.
  optimizer = tf.optimizers.Adam(learning_rate=lr)
  agent = agent_lib.Agent(
      '0', trpo_lib.TRPOActor(
          env, optimizer, model_scope='trpo_' + str(i),
          batch_size=64, num_epoch=10, future_discount=0.99,
          kl_targ=0.01, beta=1.0, lam=0.95, seed=0,
      ), init_exp=0.5, final_exp=0.0, anneal_steps=1,
      critic=critic_lib.Critic(env.state_dim, 200, seed=0)
  )

  # Set up clients.
  svfs = []
  svf_ms = []
  for i in range(num_total_clients):
    seed = int(i * 1e4)
    env = envs[i]
    client = client_lib.Client(i, 0, agent, env, num_test_epochs=2)
    client.enable_svf(5e5)
    svf_m = defaultdict(float)
    paths, episode_rewards = client.rollout(
        client.svf_n_timestep, client.env.get_parallel_envs())
    svf, svf_m = svf_lib.find_svf(-1, paths, svf_m)
    logging.error('svf shape %s, l2 norm: %.10e' % (
        svf.shape, np.linalg.norm(svf / np.sum(svf), ord=2)))
    svf_ms.append(svf_m)
  tmp = []
  tmp.extend([k for svf_m in svf_ms for k in svf_m.keys()])
  print(len(tmp))
  full_keys = list(set(tmp))
  print(len(full_keys))
  svfs = np.zeros(shape=(num_total_clients, len(full_keys),))
  for i, svf_m in enumerate(svf_ms):
    for j, k in enumerate(full_keys):
      svfs[i][j] += svf_m.get(k, 0.0)
  np.save('./svfs.npy', svfs)

  from sklearn.metrics.pairwise import cosine_similarity
  for i in cosine_similarity(svfs, svfs):
    print(['%.5f' % f for f in i])


if __name__ == "__main__":
  app.run(main)
