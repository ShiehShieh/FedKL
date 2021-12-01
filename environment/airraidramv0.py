import gym
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import math
from gym import spaces, logger
from gym.utils import seeding


class AirRaidRamV0(object):
  def __init__(self, seed=None, initial_pos_range=[-0.05, 0.05]):
    # Create environment meta.
    self.env = gym.make('AirRaid-ram-v0')
    self.state_dim   = self.env.observation_space.shape[0]
    self.num_actions = self.env.action_space.n
    self.is_continuous = False
    if seed is not None:
      self.env.seed(seed)

    # Dataset.
    state = self.env.reset()
    self.env_sample = {
        'observations': [[state.tolist()]],
        'actions': [0],
        'seq_mask': [0],
        'reward': [[0]],
        'dfr': [0],
    }
    self.output_types={
        'observations': tf.dtypes.float32,
        'actions': tf.dtypes.int32,
        'seq_mask': tf.dtypes.int32,
        'reward': tf.dtypes.float32,
        'dfr': tf.dtypes.float32,
    }
    self.output_shapes={
        'observations': [None, self.state_dim],
        'actions': [None],
        'seq_mask': [None],
        'reward': [None, 1],
        'dfr': [None],
    }

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
