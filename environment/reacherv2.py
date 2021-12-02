import numpy as np
import tensorflow as tf

import math
from gym.spaces import Box, Discrete

from gym.envs.mujoco import ReacherEnv
from gym.wrappers import TimeLimit

# import model.utils.vec_env as vec_env_lib
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


class ReacherV2(object):
  def __init__(self, seed=None,
               qpos_high_low=[[-0.2, 0.2], [-0.2, 0.2]],
               qvel_high_low=[-0.005, 0.005], action_noise=np.zeros(2)):
    self.seed = seed
    # Parallel envs for fast rollout.
    def make_env(seed):
      def _f():
        env = TimeLimit(
            CustomizedReacherEnv(
                qpos_high_low, qvel_high_low, action_noise),
            max_episode_steps=50)
        env.seed(seed)
        return env
      return _f

    # Warmup and make sure subprocess is ready, if any.
    self.make_env = make_env
    self.env = DummyVecEnv([make_env(seed)])
    self.env.reset()

    # Create environment meta.
    env = make_env(0)()
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


class CustomizedReacherEnv(ReacherEnv):
    def __init__(self, qpos_high_low=[[-0.2, 0.2], [-0.2, 0.2]],
                 qvel_high_low=[-0.005, 0.005], action_noise=np.zeros(2)):
        ReacherEnv.__init__(self)
        self.qpos_high_low = qpos_high_low
        self.qvel_high_low = qvel_high_low
        self.action_noise = action_noise
        # NOTE(XIE,Zhijie): Not a good practice though. If there is internal
        # dependence on self.step(action), we might be in trouble.
        self.step_ = self.step
        self.step = lambda action: self.step_(action + self.action_noise)

    def reset_model(self):
      qpos = (
          self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
          + self.init_qpos
      )
      while True:
        x = self.np_random.uniform(
            low=self.qpos_high_low[0][0],
            high=self.qpos_high_low[0][1], size=1)[0]
        y = self.np_random.uniform(
            low=self.qpos_high_low[1][0],
            high=self.qpos_high_low[1][1], size=1)[0]
        self.goal = np.array([x, y])
        if np.linalg.norm(self.goal) < 0.2:
          break
      qpos[-2:] = self.goal
      qvel = self.init_qvel + self.np_random.uniform(
          low=self.qvel_high_low[0], high=self.qvel_high_low[1],
          size=self.model.nv,
      )
      qvel[-2:] = 0
      self.set_state(qpos, qvel)
      return self._get_obs()
