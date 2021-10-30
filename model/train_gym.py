from __future__ import absolute_import, division, print_function

from absl import app, flags, logging

import gym
import queue
import random
import threading
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
# from collections import defaultdict

import model.rl.agent.agent as agent_lib
import model.rl.agent.ops as ops_lib

FLAGS = flags.FLAGS

flags.DEFINE_string("op", "Train", "Train or Test?")
flags.DEFINE_integer("batch_size", 32, "Sample size for one batch.")
flags.DEFINE_integer("num_epoches", 1, "Maximum number of training epoches.")
flags.DEFINE_integer("max_train_steps", None,
                     "If not None, run at most this many steps.")
flags.DEFINE_integer("max_seq_len", 256, "Maximum sequence length.")
flags.DEFINE_integer("init_action", -1, "The initial action taken.")


def mountain_car(env, mode, t):
  # Assuming that our behaviour policy will push the car to the left hill,
  # and then push it right up to the right hill.
  if mode < 0.1:
    action = env.action_space.sample()
    # elif mode < 0.6:
  else:
    rnd = random.random()
    if t < 25:
      if rnd < 0.90:
        action = 2
      elif rnd < 0.95:
        action = 0
      else:
        action = 1
    elif t < 50:
      if rnd < 0.90:
        action = 0
      elif rnd < 0.95:
        action = 2
      else:
        action = 1
    else:
      if rnd < 0.90:
        action = 2
      elif rnd < 0.95:
        action = 0
      else:
        action = 1
  # elif mode < 0.8:
  #   rnd = random.random()
  #   if t < 50:
  #     if rnd < 0.95:
  #       action = 2
  #     elif rnd < 0.98:
  #       action = 0
  #     else:
  #       action = 1
  #   else:
  #     if rnd < 0.95:
  #       action = 0
  #     elif rnd < 0.98:
  #       action = 2
  #     else:
  #       action = 1
  # else:
  #   rnd = random.random()
  #   if t < 100:
  #     if rnd < 0.95:
  #       action = 0
  #     elif rnd < 0.98:
  #       action = 2
  #     else:
  #       action = 1
  #   else:
  #     if rnd < 0.95:
  #       action = 2
  #     elif rnd < 0.98:
  #       action = 0
  #     else:
  #       action = 1
  return action


def taxi_v3(env, observation, r_table, u_table):
  action = random.randint(0, 5)
  # if r_table[observation][action] == -10:
  #   if random.random() < 0.9:
  #     action = random.randint(0, 3)
  if u_table[observation][action]:
    if random.random() < 0.9:
      n_action = random.randint(0, 5)
      while n_action == action:
        n_action = random.randint(0, 5)
      action = n_action
  return action


def gen_trajectories_for_taxi_v3():
  trajectories = []
  with open('/Users/lls/lls/src/git.llsapp.com/lq/bellman-develop/experimental/jayxie/taxi_v3_trajectories.tsv', 'r') as fp:
    seen = set()
    m = {}
    for i in range(500):
      for j in range(6):
        m[(i, j)] = 0
    for line in fp:
      if line in seen:
        continue
      seen.add(line)
      line = line.strip('\n').split('\t')
      trajectory = []
      reverse_idx = {}
      for i, e in enumerate(line):
        if i != 0 and e == line[i - 1]:
          continue
        # if e in reverse_idx:
        #   trajectory = trajectory[:reverse_idx[e] + 1]
        #   continue
        reverse_idx[e] = len(trajectory)
        event = e.split(',')
        event = [int(event[0]), int(event[1]), float(event[2]), int(event[3])]
        # if event[0] == event[-1] and random.random() < 0.9:
        #   continue
        # if event[0] == event[-1]:
        #   event[2] = -10
        trajectory.append(event)
        if len(trajectory) == 200:
          break
      last_reward = trajectory[-1][2]
      if last_reward == 20.0 or random.random() < 0.05:
        trajectories.append(trajectory)
      # elif random.random() < 0.01:
      #   trajectories.append(trajectory)
      # if len(trajectory) > 25:
      #   continue
      # trajectories.append(trajectory)
      if len(trajectories) % 10000 == 0:
        logging.error(len(trajectories))
      if len(trajectories) == 500000:
        break
    logging.error(len(trajectories))
    logging.error(set([t[-1][2] for t in trajectories]))
  # Stat.
  lens = []
  total_e = 0
  same_e = 0
  for t in trajectories:
    lens.append(len(t))
    total_e += len(t)
    for e in t:
      m[(e[0], e[1])] += 1
      if e[0] == e[-1]:
        same_e += 1
  logging.error(total_e)
  logging.error(same_e)
  m = np.array(list(m.values()))
  m = (m / np.sum(m)).tolist()
  logging.error(m)
  logging.error(len(m))
  logging.error(len([i for i in m if i == 0.0]))
  logging.error(np.min(lens))
  logging.error(np.max(lens))
  logging.error(np.mean(lens))
  return trajectories, np.max(lens)


def gen_trajectories_for_frozen_lake_4_4():
  trajectories = []
  with open('/Users/lls/lls/src/git.llsapp.com/lq/bellman-develop/experimental/jayxie/frozen_lake_4_4_trajectories.tsv', 'r') as fp:
    seen = set()
    for line in fp:
      if line in seen:
        continue
      seen.add(line)
      line = line.strip('\n').split('\t')
      trajectory = []
      for i, e in enumerate(line):
        event = e.split(',')
        event = [int(event[0]), int(event[1]), float(event[2]), int(event[3])]
        reward = event[2]
        # if reward != 1:
        #   if i == len(line) - 1:
        #     reward = -1
        #   else:
        #     reward = -0.1
        event[2] = reward
        trajectory.append(event)
      # last_reward = trajectory[-1][2]
      # if last_reward == 1 or random.random() < 0.03:
      # 70% of trajectories end with obs:5.
      trajectories.append(trajectory)
      if len(trajectories) == 50000:
        break
    logging.error(len([t[-1][2] for t in trajectories if t[-1][2] == 1]))
    logging.error(len([t[-1][2] for t in trajectories if t[-1][2] == -1]))
    logging.error(len(trajectories))
    logging.error(set([t[-1][2] for t in trajectories]))
  lens = []
  for t in trajectories:
    lens.append(len(t))
  return trajectories, np.max(lens)


def gen_trajectories(env):
  # trajectory = [[o0, a0, r0, o1], [o1, a1, r1, o2], ...]
  # Create trajectory for training.
  # rewards = []
  trajectories = []

  trajectories, max_seq_len = gen_trajectories_for_frozen_lake_4_4()
  # trajectories, max_seq_len = gen_trajectories_for_taxi_v3()

  # r_table = defaultdict(lambda: defaultdict(int))
  # u_table = defaultdict(lambda: defaultdict(bool))
  # for i in range(1000000):
  #   episode_reward = []
  #   trajectory = []
  #   observation = env.reset()
  #   # mode = random.random()
  #   for t in range(200):
  #     # env.render()

  #     event = [observation]

  #     # action = mountain_car(env, mode, t)

  #     # action = taxi_v3(env, observation, r_table, u_table)

  #     action = env.action_space.sample()

  #     next_observation, reward, done, info = env.step(action)
  #     if False:
  #       r_table[observation][action] = reward
  #       if next_observation == observation:
  #         u_table[observation][action] = True

  #     observation = next_observation
  #     event.extend([action, reward, observation])
  #     trajectory.append(event)
  #     episode_reward.append(reward)
  #     if done:
  #       # trajectories.append(trajectory)
  #       logging.info("#{}: Episode finished after {} timesteps, got {} reward.".format(i + 1, t + 1, np.sum(episode_reward)))
  #       break
  #   trajectories.append(trajectory)
  # with open('/Users/lls/lls/src/git.llsapp.com/lq/bellman-develop/experimental/jayxie/taxi_v3_trajectories.tsv', 'w') as fp:
  #   for t in trajectories:
  #     fp.write('\t'.join(','.join(map(str, e)) for e in t))
  #     fp.write('\n')
  # exit(0)

  o = [[t[0] for t in tr] for tr in trajectories]
  a = [[t[1] for t in tr] for tr in trajectories]
  r = [[[t[2]] for t in tr] for tr in trajectories]
  n = [[t[3] for t in tr] for tr in trajectories]
  return trajectories, o, a, r, n, max_seq_len


def build_dataset(env, agent, batch_size, num_epoches, max_seq_len, observation_channel, is_training=True):
  if is_training:
    trajectories, o, a, r, n, msl = gen_trajectories(env)
    max_seq_len = msl
    logging.error('#Trajectory: %d' % (len(trajectories)))
    logging.error('max_seq_len: %d' % (msl))
  obs_dim = 16
  eye = np.eye(obs_dim)

  def encode(n):
    return n
    # eye1 = np.eye(5)
    # eye2 = np.eye(4)
    # out = np.concatenate([eye1[l] if i != 3 else eye2[l]
    #                       for i, l in enumerate(list(env.decode(n)))], axis=-1)
    # return out

    # return n

    return eye[n]

  def gen():
    if is_training:
      for i in range(len(trajectories)):
        padded_seq_len = max_seq_len - len(o[i])
        # logging.error({
        #     # 'state_embedding': np.expand_dims(np.argmax(o[i], axis=-1), axis=-1).tolist() + [[0]] * padded_seq_len,
        #     'state_embedding': o[i] + [[0]] * padded_seq_len,
        #     'action_id': a[i] + [0] * padded_seq_len,
        #     'reward': r[i] + [[0]] * padded_seq_len,
        #     'seq_mask': [1] * len(o[i]) + [0] * padded_seq_len,
        # })
        yield {
            'state_embedding': [encode(oo) for oo in o[i]] + [0 * obs_dim] * padded_seq_len,
            'action_id': a[i] + [0] * padded_seq_len,
            'reward': r[i] + [[0]] * padded_seq_len,
            'next_state_embedding': [encode(nn) for nn in n[i]] + [0 * obs_dim] * padded_seq_len,
            'seq_mask': [1] * len(o[i]) + [0] * padded_seq_len,
            'label_ctx': [[1.0]] * len(o[i]) + [[0]] * padded_seq_len,
            'hidden_state': np.array([0.0] * agent.params.actor_params.u_v),
            'all_action_ids': list(range(4)),
        }
    else:
      while True:
        padded_seq_len = max_seq_len - 1
        observation, action, user_state, next_observation = observation_channel.get()
        user_state = np.squeeze(user_state)
        # logging.error(user_state)
        # logging.error({
        #     'state_embedding': [np.argmax(observation)] + [[0]] * padded_seq_len,
        #     'action_id': [action] + [0] * padded_seq_len,
        #     'seq_mask': [1] + [0] * padded_seq_len,
        #     'hidden_state': user_state,
        # })
        yield {
            'state_embedding': [encode(observation)] + [0 * obs_dim] * padded_seq_len,
            'action_id': [action if action != -1 else 0] + [0] * padded_seq_len,
            'reward': [[0]] + [[0]] * padded_seq_len,
            'next_state_embedding': [encode(next_observation)] + [0 * obs_dim] * padded_seq_len,
            'seq_mask': [1 if action != -1 else 0] + [0] * padded_seq_len,
            'label_ctx': [[1.0]] + [[0]] * padded_seq_len,
            'hidden_state': user_state,
            'all_action_ids': list(range(4)),
        }

  dataset = tf.data.Dataset.from_generator(
      gen, {
          'state_embedding': tf.int32,
          'action_id': tf.int32,
          'reward': tf.float32,
          'next_state_embedding': tf.int32,
          'seq_mask': tf.int32,
          'label_ctx': tf.float32,
          'hidden_state': tf.float32,
          'all_action_ids': tf.int32,
      }, {
          'state_embedding': tf.TensorShape([None]),
          'action_id': tf.TensorShape([None]),
          'reward': tf.TensorShape([None, 1]),
          'next_state_embedding': tf.TensorShape([None]),
          'seq_mask': tf.TensorShape([None]),
          'label_ctx': tf.TensorShape([None, 1]),
          'hidden_state': tf.TensorShape([agent.params.actor_params.u_v]),
          'all_action_ids': tf.TensorShape([4]),
      },
  )
  if not is_training:
    batch_size = 1
    num_epoches = 1
  dataset = dataset.repeat(num_epoches).batch(batch_size)
  if is_training:
    dataset = dataset.shuffle(FLAGS.batch_size * 10)
  # epoch_counter = tf.data.Dataset.range(num_epoches)
  # dataset = epoch_counter.flat_map(
  #     lambda i: tf.data.Dataset.zip((tf.data.Dataset.from_tensors(i).repeat(), dataset))
  # )
  return dataset


def run_test(env, agent, g, inputs, test_init_op, observation_channel, action_channel, sess):
  x = threading.Thread(target=agent.test, args=(g, inputs, test_init_op, action_channel, sess))
  x.start()

  rewards = []
  for _ in range(1):
    observation = env.reset()
    episode_reward = []
    for t in range(150):
      env.render()
      user_state = [None]
      if t == 0:
        try:
          action, h_state = action_channel.get_nowait()
          user_state[0] = h_state
        except queue.Empty:
          action = FLAGS.init_action
          user_state[0] = np.array([[0.0] * agent.params.actor_params.u_v])
          if action == -1:
            event = [observation, -1, np.array([[0.0] * agent.params.actor_params.u_v]), observation]
            observation_channel.put(event)
            action, h_state = action_channel.get()
            user_state[0] = h_state
      else:
        action, h_state = action_channel.get()
        user_state[0] = h_state
      # decoded_s = list(env.decode(observation))
      decoded_s = None
      # if decoded_s[:3] == [0, 0, 0] or decoded_s[:3] == [0, 4, 1] or decoded_s[:3] == [4, 0, 2] or decoded_s[:3] == [4, 3, 3]:
      #   action = 4
      # if decoded_s[:3] == [0, 0, 4]:
      #   action = 2
      next_observation, reward, done, info = env.step(action)
      event = [observation, action, user_state[0], next_observation]
      observation_channel.put(event)
      observation = next_observation
      logging.error('%s, %s, %s' % (action, reward, decoded_s))
      logging.error('')
      episode_reward.append(reward)
      if done:
        env.render()
        logging.info("Episode finished after {} timesteps, got {} reward.".format(t + 1, np.sum(episode_reward)))
        if reward == 1.0:
          exit(0)
        break
    logging.error(np.sum(episode_reward))
    rewards.append(np.sum(episode_reward))
  x.join()
  logging.error(np.mean(rewards))


def main(_):
  # Create environment.
  env = gym.make('MountainCar-v0')
  env = gym.make('Taxi-v3')
  env = gym.make('FrozenLake-v0', is_slippery=False)
  # Create agent.
  agent = agent_lib.Agent(agent_lib.Params(env.action_space.n, ops_lib.Op.TRAINING))
  agent.params.actor_params.top_k = 1

  # Frozen lake 4x4.
  agent.params.actor_params.lr = 4.0e-04
  agent.params.actor_params.future_discount = 0.999
  agent.params.actor_params.u_u = 16
  agent.params.actor_params.u_v = 16
  agent.params.actor_params.normalize_reward = True
  agent.params.actor_params.normalize_reward = False
  agent.params.actor_params.importance_weight_cap = 10.0
  agent.params.actor_params.context_mode = 'CONCATENATION'
  agent.params.actor_params.context_mode = 'LATENT_CROSS'

  # # Taxi-V3.
  # agent.params.actor_params.lr = 4.0e-04
  # agent.params.actor_params.future_discount = 0.99
  # agent.params.actor_params.u_u = 64
  # agent.params.actor_params.u_v = 128
  # agent.params.actor_params.beta_unit_size = 256
  # agent.params.actor_params.normalize_reward = True

  # Dataset.
  action_channel = queue.Queue(maxsize=1)
  observation_channel = queue.Queue(maxsize=1)
  # Training.
  if FLAGS.op == "Train":
    train_dataset = build_dataset(env, agent, FLAGS.batch_size, FLAGS.num_epoches, FLAGS.max_seq_len, observation_channel, True)
    it = tfv1.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    inputs = it.get_next()
    train_init_op = it.make_initializer(train_dataset, name='train_iter')
    agent.train(inputs, train_init_op, FLAGS.max_train_steps, None)
  # Testing.
  # tf.keras.backend.clear_session()
  if FLAGS.op == "Test":
    agent.params.actor_params.max_seq_len = 10  # For testing.
    agent.params.mode = ops_lib.Op.TESTING
    test_dataset = build_dataset(env, agent, FLAGS.batch_size, FLAGS.num_epoches, agent.params.actor_params.max_seq_len, observation_channel, False)
    it = tfv1.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
    inputs = it.get_next()
    test_init_op = it.make_initializer(test_dataset, name='test_iter')
    run_test(env, agent, tfv1.get_default_graph(), inputs, test_init_op, observation_channel, action_channel, None)

  env.close()


if __name__ == "__main__":
  app.run(main)
