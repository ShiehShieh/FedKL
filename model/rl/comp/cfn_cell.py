# -*- coding: utf-8 -*-

# Copyright (C) 2017 by Akira TAMAMORI

# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.keras import initializers


class CFNCell(tf.keras.layers.Layer):
    """ Chaos-Free Network (CFN).

       Thomas Laurent and James von Brecht,
       "A recurrent neural network without chaos,"
       https://arxiv.org/abs/1612.06212
    """

    def __init__(self, state_size, **kwargs):
        super(CFNCell, self).__init__(**kwargs)
        self._state_size = state_size
        self.state_size = [self._state_size]
        self.output_size = self._state_size
        with tfv1.variable_scope('cfn_cell', reuse=tfv1.AUTO_REUSE):
          with tfv1.variable_scope("ForgetGate"):
            self.theta_u = tf.keras.layers.Dense(
                units=self._state_size,
                name='theta_u',
                kernel_initializer=initializers.RandomUniform(-0.07, 0.07),
                kernel_regularizer=tf.keras.regularizers.l2(0.1),
                use_bias=False,
            )
            self.theta_w = tf.keras.layers.Dense(
                units=self._state_size,
                name='theta_w',
                kernel_initializer=initializers.RandomUniform(-0.07, 0.07),
                bias_initializer=initializers.Constant(1),
                kernel_regularizer=tf.keras.regularizers.l2(0.1),
                use_bias=True,
            )
          with tfv1.variable_scope("InputGate"):
            self.eta_u = tf.keras.layers.Dense(
                units=self._state_size,
                name='eta_u',
                kernel_initializer=initializers.RandomUniform(-0.07, 0.07),
                kernel_regularizer=tf.keras.regularizers.l2(0.1),
                use_bias=False,
            )
            self.eta_w = tf.keras.layers.Dense(
                units=self._state_size,
                name='eta_w',
                kernel_initializer=initializers.RandomUniform(-0.07, 0.07),
                kernel_regularizer=tf.keras.regularizers.l2(0.1),
                bias_initializer=initializers.Constant(-1),
                use_bias=True,
            )
          with tfv1.variable_scope("Input"):
            self.wx_w = tf.keras.layers.Dense(
                units=self._state_size,
                name='wx_w',
                kernel_initializer=initializers.RandomUniform(-0.07, 0.07),
                kernel_regularizer=tf.keras.regularizers.l2(0.1),
                use_bias=False,
            )

    def call(self, inputs, states, scope=None):
      state = states[0]
      theta = tfv1.sigmoid(self.theta_u(state) + self.theta_w(inputs))
      eta = tfv1.sigmoid(self.eta_u(state) + self.eta_w(inputs))
      Wx = self.wx_w(inputs)
      h = theta * tfv1.tanh(state) + eta * tfv1.tanh(Wx)

      # hidden output and state
      return h, h
