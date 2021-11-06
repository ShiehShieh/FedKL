from __future__ import absolute_import, division, print_function

from absl import app, flags, logging

from collections import defaultdict

import time
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

tfv1.disable_eager_execution()

import client.client as client_lib
import env.halfcheetahv2 as halfcheetahv2_lib
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
  num_total_clients = 10
  for i in range(num_total_clients):
    seed = int(i * 1e4)
    interval = 0.01
    # Little heterogeneity.
    """ 2e5 timesteps per client. Cosine similarity of svf:
    ['1.00000', '0.74618', '0.77991', '0.03640', '0.00092', '0.00086', '0.00063', '0.00058', '0.00092', '0.00061']
    ['0.74618', '1.00000', '0.75252', '0.03491', '0.00067', '0.00146', '0.00099', '0.00075', '0.00072', '0.00091']
    ['0.77991', '0.75252', '1.00000', '0.03655', '0.00088', '0.00109', '0.00066', '0.00074', '0.00095', '0.00074']
    ['0.03640', '0.03491', '0.03655', '1.00000', '0.47447', '0.00099', '0.00067', '0.00063', '0.00082', '0.00055']
    ['0.00092', '0.00067', '0.00088', '0.47447', '1.00000', '0.00384', '0.00104', '0.00095', '0.00130', '0.00068']
    ['0.00086', '0.00146', '0.00109', '0.00099', '0.00384', '1.00000', '0.03816', '0.00142', '0.00141', '0.00097']
    ['0.00063', '0.00099', '0.00066', '0.00067', '0.00104', '0.03816', '1.00000', '0.44469', '0.25660', '0.00296']
    ['0.00058', '0.00075', '0.00074', '0.00063', '0.00095', '0.00142', '0.44469', '1.00000', '0.46878', '0.00468']
    ['0.00092', '0.00072', '0.00095', '0.00082', '0.00130', '0.00141', '0.25660', '0.46878', '1.00000', '0.51608']
    ['0.00061', '0.00091', '0.00074', '0.00055', '0.00068', '0.00097', '0.00296', '0.00468', '0.51608', '1.00000']
    """
    x_left = -0.2 + interval * (4.0 / 3.0) * 3 * i
    # Severe heterogeneity.
    """ 2e5 timesteps per client. Cosine similarity of svf:
    ['1.00000', '0.00272', '0.00206', '0.00102', '0.00074', '0.00099', '0.00031', '0.00117', '0.00044', '0.00125']
    ['0.00272', '1.00000', '0.00169', '0.00096', '0.00064', '0.00097', '0.00035', '0.00111', '0.00050', '0.00113']
    ['0.00206', '0.00169', '1.00000', '0.00062', '0.00057', '0.00075', '0.00028', '0.00078', '0.00041', '0.00087']
    ['0.00102', '0.00096', '0.00062', '1.00000', '0.77995', '0.00087', '0.00048', '0.00063', '0.00062', '0.00070']
    ['0.00074', '0.00064', '0.00057', '0.77995', '1.00000', '0.00107', '0.00057', '0.00093', '0.00056', '0.00058']
    ['0.00099', '0.00097', '0.00075', '0.00087', '0.00107', '1.00000', '0.00118', '0.00116', '0.00079', '0.00076']
    ['0.00031', '0.00035', '0.00028', '0.00048', '0.00057', '0.00118', '1.00000', '0.00283', '0.00067', '0.00046']
    ['0.00117', '0.00111', '0.00078', '0.00063', '0.00093', '0.00116', '0.00283', '1.00000', '0.00283', '0.00066']
    ['0.00044', '0.00050', '0.00041', '0.00062', '0.00056', '0.00079', '0.00067', '0.00283', '1.00000', '0.00064']
    ['0.00125', '0.00113', '0.00087', '0.00070', '0.00058', '0.00076', '0.00046', '0.00066', '0.00064', '1.00000']
    """
    x_left = -0.5 + interval * (10.0 / 3.0) * 3 * i
    x_right = x_left + interval
    env = halfcheetahv2_lib.HalfCheetahV2(
        seed=seed, qpos_high_low=[x_left, x_right],
        qvel_high_low=[-0.005, 0.005], parallel=parallel)
    # for j in range(1000):
    #   env.render()
    #   time.sleep(0.1)
    #   env.step(env.env.action_space.sample())
    # exit(0)
    logging.error([x_left, x_right])
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
        i, 0, agent, env, num_test_epochs=2, filt=True, extra_features=[])
    clients.append(client)
    fl.register(client)
  svfs, _ = fl.get_state_visitation_frequency(clients, logging.error)
  np.save('./svfs.npy', svfs)

  from sklearn.metrics.pairwise import cosine_similarity
  for i in cosine_similarity(svfs, svfs):
    print(['%.5f' % f for f in i])


if __name__ == "__main__":
  app.run(main)
