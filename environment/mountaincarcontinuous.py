import numpy as np
import tensorflow as tf

import math
from gym.spaces import Box, Discrete

from gym.envs.classic_control import Continuous_MountainCarEnv
from gym.wrappers import TimeLimit

# import model.utils.vec_env as vec_env_lib
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


def generate_mountaincarcontinuous_heterogeneity(i, htype):
  out = [-0.6, -0.4]
  action_noise = np.zeros(shape=(1,))
  if htype == 'iid':
    pass
  if htype in ['init-state', 'both']:
    raise NotImplementedError
  if htype in ['dynamics', 'both']:
    # action_noise = np.clip(np.random.normal(0.0, 0.1, 2), -1.0, 1.0)
    # action_noise = np.clip(np.random.normal(0.0, 0.2, 2), -1.0, 1.0)
    # action_noise = np.clip(np.random.normal(0.0, 0.4, 2), -1.0, 1.0)
    action_noise = np.clip(np.random.normal(0.0, 0.8, 1), -1.0, 1.0)
    # action_noise = np.array([0.0, 0.0])
  if out[0] > out[1]:
    raise NotImplementedError
  return out, action_noise


def wrap_vector_envs(env):
  return env

  env.step_ = env.step
  env.reset_ = env.reset

  def new_step(action):
    obs, r, d, i = env.step_(action)
    new_obs = []
    for o in obs:
      new_obs.append(remove_state_about_target_position(o))
    obs = np.array(new_obs)
    return obs, r, d, i

  def new_reset():
    obs = env.reset_()
    new_obs = []
    for o in obs:
      new_obs.append(remove_state_about_target_position(o))
    obs = np.array(new_obs)
    return obs

  env.step = new_step
  env.reset = new_reset
  return env


def remove_state_about_target_position(state):
  # state = np.concatenate([state[0:4], state[6:8]], axis=0)
  state = np.concatenate([state[0:2], state[4:10]], axis=0)
  return state


class MountianCarContinuous(object):
  def __init__(self, seed=None, qpos_low_high=[-0.6, -0.4],
               action_noise=np.zeros(1)):
    self.seed = seed
    # Parallel envs for fast rollout.
    def make_env(seed):
      def _f():
        return gym.make('MountainCarContinuous-v0')
        env = TimeLimit(
            CustomizedMCCEnv(qpos_low_high, action_noise),
            max_episode_steps=999)
        env.seed(seed)
        return env
      return _f

    # Warmup and make sure subprocess is ready, if any.
    self.make_env = make_env
    self.env = DummyVecEnv([make_env(seed)])
    self.env = wrap_vector_envs(self.env)
    self.env.reset()

    # Create environment meta.
    env = make_env(0)()
    # env = wrap_vector_envs(env)
    # self.state_dim = 8  # self.env.observation_space.shape[0]
    self.state_dim = self.env.observation_space.shape[0]
    if isinstance(self.env.action_space, Box):
        self.num_actions = self.env.action_space.shape[0]
    elif isinstance(self.env.action_space, Discrete):
        self.num_actions = self.env.action_space.n
    self.is_continuous = True
    # Dataset.
    # state = env.reset()
    self.env_sample = {
        'observations': [[np.zeros(shape=self.state_dim)]],
        'actions': [np.zeros(shape=self.num_actions)],
        'seq_mask': [0],
        'reward': [[0]],
        'dfr': [0],
    }
    self.output_types={
        'observations': tf.dtypes.float32,
        'actions': tf.dtypes.float32,
        'seq_mask': tf.dtypes.int32,
        'reward': tf.dtypes.float32,
        'dfr': tf.dtypes.float32,
    }
    self.output_shapes={
        'observations': [None, self.state_dim],
        'actions': [None, self.num_actions],
        'seq_mask': [None],
        'reward': [None, 1],
        'dfr': [None],
    }
    env.close()

  def get_single_envs(self):
    return self.env

  def get_parallel_envs(self, parallel):
    # NOTE(XIE,Zhijie): We are using the same seed every time we create these
    # subprocess. It is OK because it will create the same behavior as long as
    # we run the same policy. And if the policy is changed, the trajectory ought
    # to be changed, too.
    #
    # Moreover, we postpone the creation of SubprocVecEnv here because it is
    # too expensive for us to keep num_total_clients * parallel subprocesses at
    # the same time.
    #
    # Also, it is the user's responsiblity to close the envs after use.
    envs = SubprocVecEnv(
        [self.make_env(self.seed + 1 + j) for j in range(parallel)],
        start_method='fork')
    envs = wrap_vector_envs(envs)
    return envs

  def is_solved(self, episode_history):
    return False

  def render(self):
    return self.env.render()

  def reset(self):
    return self.env.reset()[0]

  def step(self, action):
    obs, r, d, i = self.env.step([action])
    return obs[0], r[0], d[0], i[0]

  def cleanup(self):
    self.env.close()


class CustomizedMCCEnv(Continuous_MountainCarEnv):
    def __init__(self, qpos_low_high=[-0.6, -0.4],
                 action_noise=np.zeros(1)):
        Continuous_MountainCarEnv.__init__(self)
        self.qpos_low_high = qpos_low_high
        self.action_noise = action_noise
        # NOTE(XIE,Zhijie): Not a good practice though. If there is internal
        # dependence on self.step(action), we might be in trouble.
        # self.step_ = self.step
        # self.step = lambda action: self.step_(action + self.action_noise)

        def reset_():
          low = self.qpos_low_high[0]
          high = self.qpos_low_high[1]
          self.state = np.array(
              [self.np_random.uniform(low=low, high=high), 0]
          )
          return np.array(self.state, dtype=np.float32)

        # self.reset = reset_
