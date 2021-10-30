#!/usr/bin/env python

from absl import logging

import time
import numpy as np
from collections import deque, defaultdict

import model.rl.comp.state_visitation_frequency as svf_lib
import model.utils.filters as filters_lib
import model.utils.utils as utils_lib


class Client(object):
  def __init__(self, cid, group, agent, env, num_test_epochs=100):
    self.cid = cid
    self.group = group
    self.agent = agent
    self.env = env
    self.num_episodes_seen = 0
    self.num_iter_seen = 0
    self.episode_history = deque(maxlen=50)
    self.obfilt = filters_lib.ZFilter(env.env.observation_space.shape, clip=5)
    self.rewfilt = filters_lib.ZFilter((), demean=False, clip=10)
    self.num_test_epochs = num_test_epochs
    self.use_svf = False

  def set_params(self, model_params):
    return self.agent.set_params(model_params)

  def get_params(self):
    return self.agent.get_params()

  def get_client_weight(self):
    return self.agent.get_num_timestep_seen()

  def sync_optimizer(self):
    return self.agent.sync_optimizer()

  def sync_old_policy(self):
    return self.agent.sync_old_policy()

  def enable_svf(self):
    self.use_svf = True

  def disable_svf(self):
    self.use_svf = False

  def cleanup(self):
    self.env.cleanup()

  def test(self):
    num_episodes = self.num_test_epochs
    episode_history = deque(maxlen=num_episodes)
    for i_episode in range(num_episodes):
      # initialize
      state = self.env.reset()
      state = self.obfilt(state)
      total_reward = 0

      for t in range(10000):
        # env.render()
        action, probs = self.agent.act(state)
        next_state, reward, done, _ = self.env.step(action)
        total_reward += reward

        next_state = self.obfilt(next_state)
        reward = self.rewfilt(reward)
        state = next_state
        if done:
          break
      episode_history.append(total_reward)

    return np.mean(episode_history)

  def path_length(self, trajectory):
    return len(trajectory['actions'])

  def rollout(self, n_timesteps):
    paths = []
    timesteps_sofar = 0
    episode_rewards = []
    while True:
      # initialize
      state = self.env.reset()
      state = self.obfilt(state)
      total_reward = 0

      # Generate trajectory.
      trajectory = defaultdict(list)
      while True:
        # env.render()
        action, probs = self.agent.epsilon_greedy(state)
        next_state, reward, done, _ = self.env.step(action)
        total_reward += reward

        next_state = self.obfilt(next_state)
        reward = self.rewfilt(reward)
        trajectory['observations'].append(state)
        trajectory['actions'].append(action)
        trajectory['reward'].append(reward)
        trajectory['next_observations'].append(next_state)
        trajectory['probs'].append(probs[0])

        state = next_state
        if done:
          break

      trajectory['dfr'] = utils_lib.discount(
          trajectory['reward'], self.agent.policy.future_discount)
      paths.append(trajectory)
      episode_rewards.append(total_reward)
      timesteps_sofar += self.path_length(trajectory)
      if timesteps_sofar > n_timesteps:
        break

    if self.agent.critic is not None:
      utils_lib.generalized_advantage_estimate(
          self.agent.value, paths, self.agent.policy.future_discount,
          self.agent.policy.lam)
    return paths, episode_rewards

  def experiment(self, num_iter, timestep_per_batch,
                 callback_before_fit=[], logger=None):
    svf_m = defaultdict(float)
    paths, episode_rewards = self.rollout(100000)
    svf, svf_m = svf_lib.find_svf(-1, paths, svf_m) \
        if self.use_svf else (np.zeros(shape=[1]), svf_m)
    if logger:
      logger('svf shape %s, l2 norm: %.10e' % (
          svf.shape, np.linalg.norm(svf / np.sum(svf), ord=2)))

    for i in range(num_iter):
      self.num_iter_seen += 1
      # rollout using current policy.
      paths, episode_rewards = self.rollout(timestep_per_batch)
      steps = utils_lib.convert_trajectories_to_steps(paths, shuffle=True)
      steps['svf'] = svf
      # Sync, cleanup, etc.
      for cb in callback_before_fit:
        cb()
      # Training policy.
      self.agent.fit(steps, logger)
      # Decrease epsilon after each episode.
      self.num_episodes_seen += 1
      self.agent.anneal_exploration(self.num_episodes_seen)

      self.episode_history.extend(episode_rewards)
      mean_rewards = np.mean(self.episode_history)

      if logger:
        logger("Client {}, Iteration {}".format(self.cid, self.num_iter_seen))
        logger("Average reward for last {} episodes: {:.2f}".format(min(len(self.episode_history), self.episode_history.maxlen), mean_rewards))
