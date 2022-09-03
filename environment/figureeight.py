import numpy as np
import tensorflow as tf

import gym
from gym import spaces
from gym.utils import seeding

from stable_baselines3.common.vec_env import VecEnv

from flow.benchmarks.figureeight1 import flow_params as flow_params_v1
from flow.benchmarks.figureeight2 import flow_params as flow_params_v2
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.utils.registry import make_create_env


combination = [
    'h', 'r', 'h', 'h', 'r', 'r', 'h', 'r', 'h', 'h', 'h', 'r', 'r', 'r',
]
vehicles = VehicleParams()
for i, c in enumerate(combination):
  if c == 'h':
    vehicles.add(
        veh_id="human_{}".format(i),
        acceleration_controller=(IDMController, {"noise": 0.2}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed", decel=1.5,
        ),
        num_vehicles=1)
  elif c == 'r':
    vehicles.add(
        veh_id="rl_{}".format(i),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        num_vehicles=1)
  else:
    pass

# for i in range(7):
#   vehicles.add(
#       veh_id="human{}".format(i),
#       acceleration_controller=(IDMController, {
#           "noise": 0.2
#       }),
#       routing_controller=(ContinuousRouter, {}),
#       car_following_params=SumoCarFollowingParams(
#           speed_mode="obey_safe_speed",
#           decel=1.5,
#       ),
#       num_vehicles=1)
#   vehicles.add(
#       veh_id="rl{}".format(i),
#       acceleration_controller=(RLController, {}),
#       routing_controller=(ContinuousRouter, {}),
#       car_following_params=SumoCarFollowingParams(
#           speed_mode="obey_safe_speed",
#           ),
#       num_vehicles=1)


flow_params_v1['sim'].print_warnings = False
flow_params_v1['sim'].seed = 0
flow_params_v1['sim'].restart_instance = True
flow_params_v1['veh'] = vehicles
_create_env_v1, _env_name_v1 = make_create_env(flow_params_v1, version=1)
_global_env_v1 = _create_env_v1()

flow_params_v2['sim'].print_warnings = False
flow_params_v2['sim'].seed = 0
flow_params_v2['sim'].restart_instance = True
_create_env_v2, _env_name_v2 = make_create_env(flow_params_v2, version=2)
_global_env_v2 = _create_env_v2()


class FlowFigureEight(VecEnv):
  def __init__(self, seed=None):
    # VecEnv.__init__(self, num_envs, observation_space, action_space)
    pass

  def _from_state(self, state):
    num_vehicle = len(state) / 2
    assert num_vehicle == self.num_vehicle, "# vehicle is not consistent: got: %s, want: %s" % (num_vehicle, self.num_vehicle)
    states = []
    for i in self.cav_idx:
      ahead = i - 1 if i > 0 else int(num_vehicle - 1)
      behind = i + 1 if i < int(num_vehicle - 1) else 0
      s = np.array(state[2 * ahead:2 * ahead + 2].tolist() + \
                   state[2 * i:2 * i + 2].tolist() + \
                   state[2 * behind:2 * behind + 2].tolist())
      states.append(s)
    return states

  def _from_reward(self, r):
    return [r for _ in range(self.num_cav)]

  def _from_done(self, done):
    return [done for _ in range(self.num_cav)]

  def _from_info(self, info):
    return [info for _ in range(self.num_cav)]

  def _to_action(self, actions):
    return np.concatenate(self.actions, axis=0)

  def reset(self):
    # NOTE(XIE,Zhijie): Remember to set restart_instance to True, or it
    # can't be reset.
    return self._from_state(self.global_env.reset())
    # NOTE(XIE,Zhijie): The reset function of flow and sumo is buggy.
    # cf. https://www.eclipse.org/lists/sumo-user/msg05429.html
    #     https://github.com/eclipse/sumo/issues/6479
    self.global_env.terminate()
    self.global_env.close()
    num_try = 0
    while num_try < 3:
      try:
        num_try += 1
        self.global_env = gym.envs.make(self.env_name)
        break
      except BlockingIOError as e:
        print('%s' % e)
        if num_try == 3:
          raise e
    return self._from_state(self.global_env.reset())

  def step_async(self, actions):
    self.actions = actions

  def step_wait(self):
    s, r, done, info = self.global_env.step(self._to_action(self.actions))
    # Sanity check. It happens that we lose some vehicles during training.
    if len(s) / 2 != self.num_vehicle:
      info['err'] = '# vehicle is not consistent: got: %s, want: %s. actions: %s. r: %s. done: %s. info: %s.' % (len(s) / 2, self.num_vehicle, self.actions, r, done, info)
      return ([[0.0 for _ in range(6)] for _ in range(self.num_vehicle)],
              self._from_reward(r), self._from_done(True),
              self._from_info(info))
    return (self._from_state(s), self._from_reward(r),
            self._from_done(done), self._from_info(info))

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


class FlowFigureEightV1(FlowFigureEight):
  def __init__(self, seed=None):
    FlowFigureEight.__init__(self)
    self.seed = seed
    self.flow_params = flow_params_v1
    self.num_vehicle = flow_params_v1['veh'].num_vehicles
    self.num_cav = flow_params_v1['veh'].num_rl_vehicles

    # create and register the environment with OpenAI Gym
    self.env_name = _env_name_v1
    self.global_env = gym.envs.make(self.env_name)
    # https://github.com/flow-project/flow/blob/master/flow/benchmarks/figureeight1.py#L25
    self.cav_idx = [1, 3, 5, 7, 9, 11, 13]
    self.num_envs = len(self.cav_idx)


class FlowFigureEightV2(FlowFigureEight):
  def __init__(self, seed=None):
    FlowFigureEight.__init__(self)
    self.seed = seed
    self.flow_params = flow_params_v2
    self.num_vehicle = flow_params_v2['veh'].num_vehicles
    self.num_cav = flow_params_v2['veh'].num_rl_vehicles

    # create and register the environment with OpenAI Gym
    self.env_name = _env_name_v2
    self.global_env = gym.envs.make(self.env_name)
    # https://github.com/flow-project/flow/blob/master/flow/benchmarks/figureeight1.py#L25
    self.cav_idx = list(range(14))
    self.num_envs = len(self.cav_idx)


_SUMO_ENVS = FlowFigureEightV1(0)
_SUMO_ENVS_STATE = _SUMO_ENVS.reset()[0]


class CustomizedCAV(object):
  """It is the same for all versions."""

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
