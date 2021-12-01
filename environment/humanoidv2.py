import gym
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import math
from gym import spaces, logger
from gym.spaces import Box, Discrete
from gym.utils import seeding


class HumanoidV2(object):
  def __init__(self, seed=None, initial_pos_range=[-0.05, 0.05]):
    # Create environment meta.
    # self.env = RewScale(gym.make('Humanoid-v2'), 0.1)
    # Parallel envs for fast rollout.
    def make_env(seed):
      def _f():
        env = gym.make('Humanoid-v2')
        env.seed(seed)
        return env
      return _f

    # Warmup and make sure subprocess is ready.
    self.env = SubprocVecEnv([make_env(seed)], start_method='fork')
    self.env.reset()
    if parallel is not None:
      self.envs = SubprocVecEnv(
          [make_env(seed + 1 + j) for j in range(parallel)],
          start_method='fork')
      self.envs.reset()

    # Create environment meta.
    env = gym.make('Humanoid-v2')
    self.state_dim   = self.env.observation_space.shape[0]
    if isinstance(self.env.action_space, Box):
        self.num_actions = self.env.action_space.shape[0]
    elif isinstance(self.env.action_space, Discrete):
        self.num_actions = self.env.action_space.n
    self.is_continuous = True
    # Dataset.
    state = env.reset()
    self.env_sample = {
        'observations': [[state.tolist()]],
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

  def get_parallel_envs(self):
    return self.envs

  def is_solved(self, episode_history):
    return False

  def render(self):
    return self.env.render()

  def reset(self):
    return self.env.reset()

  def step(self, action):
    return self.env.step(action)

  def cleanup(self):
    self.env.close()


class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale
