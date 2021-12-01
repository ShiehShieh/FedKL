"""
State-Value Function
Written by Patrick Coady (pat-coady.github.io)
"""
from __future__ import absolute_import, division, print_function

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import numpy as np

import math
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

tfv1.disable_v2_behavior()
# Need to enable TFv2 control flow with support for higher order derivatives
# in keras LSTM layer.
tfv1.enable_control_flow_v2()


LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01


class Critic(object):
    """ NN-based state-value function """
    def __init__(self, obs_dim, hid1_mult, seed=None):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        self.seed = seed
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.epochs = 30
        self.lr = None  # learning rate set in _build_model()
        self.model = self._build_model()

    def set_params(self, model_params=None):
      if model_params is None:
        return
      self.model.set_weights(model_params)

    def get_params(self, var_list=None):
      return self.model.get_weights()

    def _build_model(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        obs = Input(shape=(self.obs_dim,), dtype='float32', name='value_network_input')
        # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
        hid1_units = 64
        hid2_units = 64
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 3e-4
        print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
              .format(hid1_units, hid2_units, 0, self.lr))
        y = Dense(
            hid1_units, activation='tanh',
            kernel_initializer=tf.keras.initializers.GlorotNormal(self.seed),
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))(obs)
        y = Dense(
            hid2_units, activation='tanh',
            kernel_initializer=tf.keras.initializers.GlorotNormal(self.seed),
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
        y = Dense(
            1, kernel_initializer=tf.keras.initializers.GlorotNormal(self.seed))(y)
        model = Model(inputs=obs, outputs=y)
        optimizer = Adam(self.lr)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def fit(self, x, y):
        """ Fit model to current data batch + previous data batch
        Args:
            x: features
            y: target
        """
        num_batches = max(x.shape[0] // 128, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.model.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        o = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=batch_size,
                       shuffle=True, verbose=0)
        y_hat = self.model.predict(x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func
        return o

    def predict(self, x):
        """ Predict method """
        return np.squeeze(self.model.predict([x]), axis=1)
