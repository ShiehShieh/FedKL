#!/usr/bin/env python

from absl import logging

import numpy as np
from collections import deque, defaultdict

import model.rl.comp.state_visitation_frequency as svf_lib
import model.utils.filters as filters_lib
import model.utils.utils as utils_lib


class Client(object):
  def __init__(
      self, cid, group, agent, env, num_test_epochs=100, filt=True,
      parallel=1, extra_features=set(['next_observations', 'probs'])):
    self.cid = cid
    self.group = group
    self.agent = agent
    self.env = env
    self.num_episodes_seen = 0
    self.num_iter_seen = 0
    self.episode_history = deque(maxlen=20)
    self.obfilt = filters_lib.IDENTITY
    self.rewfilt = filters_lib.IDENTITY
    if filt:
      self.obfilt = filters_lib.ZFilter(
          env.env.observation_space.shape, clip=5)
      self.rewfilt = filters_lib.ZFilter((), demean=False, clip=10)
    self.num_test_epochs = num_test_epochs
    self.use_svf = False
    self.extra_features = extra_features
    self.parallel = parallel

  def set_params(self, model_params):
    return self.agent.set_params(model_params)

  def get_params(self):
    return self.agent.get_params()

  def reset_client_weight(self):
    return self.agent.reset_num_timestep_seen()

  def get_client_weight(self):
    return self.agent.get_num_timestep_seen()

  def sync_optimizer(self):
    return self.agent.sync_optimizer()

  def sync_old_policy(self):
    return self.agent.sync_old_policy()

  def sync_anchor_policy(self):
    return self.agent.sync_anchor_policy()

  def cleanup(self):
    self.env.cleanup()

  def test(self):
    parallel = self.num_test_epochs
    if self.num_test_epochs > 20:
      parallel = 20
    envs = self.env.get_parallel_envs(parallel)
    _, episode_rewards = self.rollout(envs, -1, self.num_test_epochs)
    envs.close()
    return np.mean(episode_rewards)

  def path_length(self, trajectory):
    return len(trajectory['actions'])

  def rollout(self, envs, n_timesteps, n_episodes):
    # Even though the input envs start from the same seed, we can still have
    # different trajectory generated if agent has been updated.
    return self.subproc_vec_env_rollout(envs, n_timesteps, n_episodes)

  def subproc_vec_env_rollout(self, envs, n_timesteps=-1, n_episodes=-1):
    if (n_timesteps < 0 and n_episodes < 0) or \
        (n_timesteps > 0 and n_episodes > 0):
      raise Exception('either n_timesteps: [%s] or n_episodes: [%s] should be larger than 0' % (n_timesteps, n_episodes))

    paths = []
    timesteps_sofar = 0
    episode_rewards = []
    while True:
      states = envs.reset()
      states = np.array([self.obfilt(state) for state in states])
      total_rewards = np.zeros(shape=(envs.num_envs,)).tolist()

      # Generate trajectory.
      trajectories = [defaultdict(list) for i in range(envs.num_envs)]
      already_done = np.zeros(shape=(envs.num_envs,), dtype=bool)
      while True:
        actions, probses = self.agent.epsilon_greedy(states)
        next_states, rewards, dones, _ = envs.step(actions)

        # Process each env seperately.
        for i, trajectory in enumerate(trajectories):
          next_states[i] = self.obfilt(next_states[i])
          if already_done[i]:
            # This env has finished its episode here.
            continue
          already_done[i] = dones[i]
          total_rewards[i] += rewards[i]
          rewards[i] = self.rewfilt(rewards[i])
          trajectory['observations'].append(states[i])
          trajectory['actions'].append(actions[i])
          trajectory['reward'].append(rewards[i])
          if 'next_observations' in self.extra_features:
            trajectory['next_observations'].append(next_states[i])
          if 'probs' in self.extra_features:
            trajectory['probs'].append(probses[i])

        states = next_states
        if np.all(dones):
          break

      episode_rewards.extend(total_rewards)
      for trajectory in trajectories:
        trajectory['dfr'] = utils_lib.discount(
            trajectory['reward'], self.agent.policy.future_discount)
        paths.append(trajectory)
        timesteps_sofar += self.path_length(trajectory)
      if n_timesteps > 0 and timesteps_sofar > n_timesteps:
        break
      if n_episodes > 0 and len(paths) > n_episodes:
        paths = paths[:n_episodes]
        break

    if self.agent.critic is not None:
      utils_lib.generalized_advantage_estimate(
          self.agent.value, paths, self.agent.policy.future_discount,
          self.agent.policy.lam)
    return paths, episode_rewards

  def enable_svf(self, svf_n_timestep=1e6):
    self.use_svf = True
    self.svf_n_timestep = svf_n_timestep

  def disable_svf(self):
    self.use_svf = False

  def get_state_visitation_frequency(self):
    svf_m = defaultdict(float)
    envs = self.env.get_parallel_envs(self.parallel)
    paths, _ = self.rollout(envs, self.svf_n_timestep, -1)
    d = svf_lib.find_discounted_svf(
        -1, paths, gamma=self.agent.policy.future_discount)
    envs.close()
    return d

  def experiment(self, num_iter, timestep_per_batch,
                 callback_before_fit=[], logger=None, norm_penalty=None):
    svf = np.zeros(shape=(1,))
    if self.use_svf and norm_penalty is None:
      d = self.get_state_visitation_frequency()
      if logger:
        logger('# state %d, l2 norm: %.10e' % (
            len(d), np.linalg.norm(d.values(), ord=2)))
    if norm_penalty is None:
      norm_penalty = np.zeros(shape=(1,))

    for i in range(num_iter):
      self.num_iter_seen += 1
      # rollout using current policy.
      paths, episode_rewards = self.rollout(
          self.env.get_single_envs(), timestep_per_batch, -1)
      steps = utils_lib.convert_trajectories_to_steps(paths, shuffle=True)
      steps['svf'] = svf
      steps['norm_penalty'] = norm_penalty
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
      logger("Client {}, Iteration {}, Weight {}".format(self.cid, self.num_iter_seen, self.get_client_weight()))
      logger("Average reward for last {} episodes: {:.2f}".format(min(len(self.episode_history), self.episode_history.maxlen), mean_rewards))
      logger("policy stat: {}".format(self.agent.stat()))

    return mean_rewards
