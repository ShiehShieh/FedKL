import gym
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import math
from gym import spaces, logger
from gym.utils import seeding


class FetchPickAndPlaceV1(object):
  def __init__(self, seed=None, initial_pos_range=[-0.05, 0.05]):
    # Create environment meta.
    self.env = gym.make('FetchPickAndPlace-v1')
    # Simply wrap the goal-based environment using FlattenDictWrapper
    # and specify the keys that you would like to use.
    state = self.env.reset()
    state = self.build_state(state)
    self.is_continuous = False
    # self.state_dim   = self.env.observation_space.shape[0]
    # self.num_actions = self.env.action_space.n
    self.state_dim   = state.shape[0]
    self.num_actions = self.env.action_space.shape[0]
    if seed is not None:
      self.env.seed(seed)

    # Dataset.
    state = self.env.reset()
    state = self.build_state(state)
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

  def build_state(self, state):
    obs = state['observation']
    achieved = state['achieved_goal']
    desired = state['desired_goal']
    return np.concatenate([obs, desired])

  def is_solved(self, episode_history):
    return False

  def render(self):
    return self.env.render()

  def reset(self):
    return self.build_state(self.env.reset())

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    substitute_goal = obs['achieved_goal'].copy()
    substitute_reward = self.env.compute_reward(
        obs['achieved_goal'], substitute_goal, info)
    # print('reward is {}, substitute_reward is {}'.format(
    #     reward, substitute_reward))
    obs = self.build_state(obs)
    return obs, reward, done, info

  def cleanup(self):
    self.env.close()
