import numpy as np
import tensorflow as tf

import gym
from gym import spaces
from gym.utils import seeding

from stable_baselines3.common.vec_env import VecEnv

from flow.benchmarks.figureeight1 import flow_params
from flow.utils.registry import make_create_env


flow_params['sim'].print_warnings = False
flow_params['sim'].seed = 0
_create_env, _env_name = make_create_env(flow_params, version=0)
_global_env = _create_env()


class FlowFigureEightV1(VecEnv):
  def __init__(self, seed=None):
    self.seed = seed
    self.flow_params = flow_params
    self.num_vehicle = flow_params['veh'].num_vehicles
    self.num_cav = flow_params['veh'].num_rl_vehicles

    # create and register the environment with OpenAI Gym
    self.global_env = gym.envs.make(_env_name)
    # https://github.com/flow-project/flow/blob/master/flow/benchmarks/figureeight1.py#L25
    self.cav_idx = [1, 3, 5, 7, 9, 11, 13]
    self.num_envs = len(self.cav_idx)

  def _from_state(self, state):
    num_vehicle = len(state) / 2
    assert num_vehicle == self.num_vehicle, "# vehicle is not consistent: got: %s, want: %s" % (num_vehicle, self.num_vehicle)
    states = []
    for i in self.cav_idx:
      ahead = i - 1 if i > 0 else (num_vehicle - 1)
      behind = i + 1 if i < (num_vehicle - 1) else 0
      s = np.array(state[2 * ahead: 2 * ahead + 2].tolist() + \
                   state[2 * i: 2 * i + 2].tolist() + \
                   state[2 * behind: 2 * behind + 2].tolist())
      states.append(s)
    return states

  def _from_reward(self, r):
    # TODO(XIE,Zhijie): Divided by self.num_cav?
    return [r for _ in range(self.num_cav)]

  def _from_done(self, done):
    return [done for _ in range(self.num_cav)]

  def _from_info(self, info):
    return [info for _ in range(self.num_cav)]

  def _to_action(self, actions):
    return np.concatenate(self.actions, axis=0)

  def reset(self):
    # NOTE(XIE,Zhijie): The reset function of flow and sumo is buggy.
    # cf. https://www.eclipse.org/lists/sumo-user/msg05429.html
    self.global_env.terminate()
    self.global_env.close()
    self.global_env = gym.envs.make(_env_name)
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


_SUMO_ENVS = FlowFigureEightV1(0)
_SUMO_ENVS_STATE = _SUMO_ENVS.reset()[0]


class CustomizedCAV(object):

  def __init__(self):
    # Create environment meta.
    state = _SUMO_ENVS_STATE
    num_vehicle = len(state) / 2
    self.state_dim = 2 * 3
    self.num_actions = 1
    self.is_continuous = True
    self.env = type('', (), {})()
    self.env.observation_space = spaces.Box(
        low=0,
        high=1,
        shape=(2 * 3, ),
        dtype=np.float32)
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


def cleanup():
  _SUMO_ENVS.close()
