import numpy as np
import tensorflow as tf

import gym
from gym import spaces
from gym.utils import seeding

from stable_baselines3.common.vec_env import VecEnv

from flow.benchmarks.figureeight1 import flow_params
from flow.utils.registry import make_create_env


class FlowFigureEightV1(VecEnv):
  def __init__(self, seed=None):
    self.seed = seed

    # create and register the environment with OpenAI Gym
    create_env, env_name = make_create_env(flow_params, version=0)
    self.global_env = create_env()
    state = self.global_env.reset()
    self.num_vehicle = len(state) / 2

  def _from_state(self, state):
    num_vehicle = len(state) / 2
    assert num_vehicle == self.num_vehicle, "# vehicle is not consistent: got: %s, want: %s" % (num_vehicle, self.num_vehicle)
    states = []
    for i in range(self.num_vehicle):
      ahead = i - 1 if i > 0 else (num_vehicle - 1)
      behind = i + 1 if i < (num_vehicle - 1) else 0
      state = np.array(state[2 * ahead: 2 * ahead + 1] + \
                       state[2 * i: 2 * i + 1] + \
                       state[2 * behind: 2 * behind + 1])
      states.append(state)
    return states

  def _from_reward(self, r):
    return [r for _ in range(self.num_vehicle)]

  def _from_done(self, done):
    return [done for _ in range(self.num_vehicle)]

  def _from_info(self, info):
    return [info for _ in range(self.num_vehicle)]

  def _to_action(self, actions):
    action = np.concatenate(self.actions, axis=0)

  def reset(self):
    return self._from_state(self.global_env.reset())

  def step_async(self, actions):
    self.actions = actions

  def step_wait(self):
    s, r, done, info = self.global_env.step(self._to_action(self.actions))
    return self._from_state(s), self._from_reward(r), self._from_done(done), self._from_info(info)

  def close(self):
    self.global_env.close()

  def seed(self, seed=None):
    self.seed = seed

  def render(self):
    raise NotImplementedError

  def env_is_wrapped(self, wrapper_class, indices=None):
    raise NotImplementedError

  def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
    raise NotImplementedError

  def get_attr(self, attr_name, indices=None):
    raise NotImplementedError

  def set_attr(self, attr_name, value, indices=None):
    raise NotImplementedError


class CustomizedCAV(object):

  def __init__(self):
    # Create environment meta.
    env = FlowFigureEightV1(0)
    state = env.reset()
    num_vehicle = len(state) / 2
    self.state_dim = 2 * 3
    self.num_actions = 1
    self.is_continuous = True
    # Dataset.
    self.env_sample = {
        'observations': [[state.tolist()]],
        'actions': [np.zeros(shape=self.num_actions)],
        'reward': [[0.0]],
    }
    self.output_types={
        'observations': tf.dtypes.float32,
        'actions': tf.dtypes.float32,
        'reward': tf.dtypes.float32,
    }
    self.output_shapes={
        'observations': [None, self.state_dim],
        'actions': [None, self.num_actions],
        'reward': [None, 1],
    }
    env.close()
