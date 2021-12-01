import gym
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import math
from gym import spaces, logger
from gym.spaces import Box, Discrete
from gym.utils import seeding

from gym import utils
from gym.envs.mujoco import Walker2dEnv
from gym.wrappers import TimeLimit


class Walker2dV2(object):
  def __init__(self, seed=None, initial_pos_range=[-0.05, 0.05]):
    # Create environment meta.
    self.env = TimeLimit(CustomizedWalker2dEnv(), max_episode_steps=1000)
    self.state_dim   = self.env.observation_space.shape[0]
    if isinstance(self.env.action_space, Box):
        self.num_actions = self.env.action_space.shape[0]
    elif isinstance(self.env.action_space, Discrete):
        self.num_actions = self.env.action_space.n
    self.is_continuous = True
    if seed is not None:
      self.env.seed(seed)

    # Dataset.
    state = self.env.reset()
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


class CustomizedWalker2dEnv(Walker2dEnv):
    def __init__(self):
        Walker2dEnv.__init__(self)

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()
