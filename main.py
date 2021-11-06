from __future__ import absolute_import, division, print_function

from absl import app, flags, logging

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

tfv1.disable_eager_execution()

import client.client as client_lib
import env.halfcheetahv2 as halfcheetahv2_lib
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
flags.DEFINE_integer("num_clients", 10, "The number of clients.")

flags.DEFINE_float("lr", "1e-4", "Learning rate.")
flags.DEFINE_float("mu", "1e-4", "Penalty coefficient for FedProx.")
flags.DEFINE_string("fed", "FedAvg", "Federated Learning Algorithm.")
flags.DEFINE_string("pg", "REINFORCE", "Policy Gradient Algorithm.")

flags.DEFINE_bool("linear", False, "Use linear layer for MLP.")
flags.DEFINE_integer("parallel", 10, "Parallelism for env rollout.")
flags.DEFINE_float("svf_n_timestep", 1e6, "The number of timestep for estimating state visitation frequency.")


def generate_heterogeneity(i):
  # 30 clients.
  interval = 0.01
  x_left = -0.5 + interval * (10.0 / 3.0) * i
  x_right = x_left + interval
  return x_left, x_right

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

  # Federated Learning Experiments.
  lr = FLAGS.lr
  mu = FLAGS.mu
  params = {
      'clients_per_round': 5,
      'num_rounds': 100,
      # The more local iteration, the more likely for FedAvg to diverge.
      'num_iter': 50,
      'timestep_per_batch': 2048,
      'max_steps': 10000,
      'eval_every': 1,
      'drop_percent': 0.0,
      'verbose': True,
      'svf_n_timestep': FLAGS.svf_n_timestep,
  }
  if FLAGS.fed == 'FedAvg':
    fl = fedavg_lib.FedAvg(**params)
    # opt_class = lambda: tf.optimizers.Adam(learning_rate=lr)
    opt_class = lambda: tf.optimizers.SGD(learning_rate=lr)
  elif FLAGS.fed == 'FedProx':
    fl = fedprox_lib.FedProx(**params)
    opt_class = lambda: pgd_lib.PerturbedGradientDescent(
        learning_rate=lr, mu=mu)
  elif FLAGS.fed == 'FedTRPO':
    fl = fedtrpo_lib.FedTRPO(**params)
    opt_class = lambda: tf.optimizers.SGD(learning_rate=lr)

  # Create env before hand for saving memory.
  envs = []
  num_total_clients = 30
  for i in range(num_total_clients):
    seed = int(i * 1e4)
    x_left, x_right = generate_heterogeneity(i)
    env = halfcheetahv2_lib.HalfCheetahV2(
        seed=seed, qpos_high_low=[x_left, x_right],
        qvel_high_low=[-0.005, 0.005])
    logging.error([x_left, x_right])
    envs.append(env)

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
          ), init_exp=0.5, final_exp=0.0, anneal_steps=1,
          critic=critic_lib.Critic(env.state_dim, 200, seed=seed)
      )
    # agent.build()

    client = client_lib.Client(
        i, 0, agent, env, num_test_epochs=10, parallel=FLAGS.parallel,
        extra_features=set([]))
    fl.register(client)

  # Start FL training.
  fl.train()


if __name__ == "__main__":
  app.run(main)
