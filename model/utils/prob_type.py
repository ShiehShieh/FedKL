import theano
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import model.utils.distributions as distributions_lib
import model.utils.utils as utils_lib


floatX = theano.config.floatX


class ProbType(object):
    def sampled_variable(self):
        raise NotImplementedError

    def prob_variable(self):
        raise NotImplementedError

    def likelihood(self, a, prob):
        raise NotImplementedError

    def loglikelihood(self, a, prob):
        raise NotImplementedError

    def kl(self, prob0, prob1):
        raise NotImplementedError

    def entropy(self, prob):
        raise NotImplementedError

    def maxprob(self, prob):
        raise NotImplementedError


class Categorical(ProbType):
    def __init__(self, n):
        self.n = n

    def sampled_variable(self):
        return T.ivector('a')

    def prob_variable(self):
        return T.matrix('prob')

    def likelihood(self, a, prob):
        return tf.gather(prob, a, batch_dims=-1)

    def loglikelihood(self, a, prob):
        return tf.math.log(self.likelihood(a, prob))

    def kl(self, prob0, prob1):
        return tf.reduce_sum(
            (prob0 * tf.math.log(
                utils_lib.stablize(prob0) / utils_lib.stablize(prob1)
            )
          ), axis=1
        )

    def entropy(self, prob0):
        return - (prob0 * T.log(prob0)).sum(axis=1)

    def sample(self, prob):
        return distributions_lib.categorical_sample(prob)

    def maxprob(self, prob):
        return tf.math.argmax(prob, axis=-1)


class DiagGauss(ProbType):
    def __init__(self, d):
        self.d = d

    def sampled_variable(self):
        return T.matrix('a')

    def prob_variable(self):
        return T.matrix('prob')

    def loglikelihood(self, a, prob):
        mean0 = prob[:, :self.d]
        std0 = prob[:, self.d:]
        # exp[ -(a - mu)^2/(2*sigma^2) ] / sqrt(2*pi*sigma^2)
        return - 0.5 * tf.reduce_sum(tf.math.square((a - mean0) / std0), axis=1) - 0.5 * tf.math.log(2.0 * np.pi) * self.d - tf.reduce_sum(tf.math.log(std0), axis=1)

    def likelihood(self, a, prob):
        return tf.math.exp(self.loglikelihood(a, prob))

    def kl(self, prob0, prob1):
        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        return tf.reduce_sum(tf.math.log(std1 / std0), axis=1) + tf.reduce_sum((tf.math.square(std0) + tf.math.square(mean0 - mean1)) / (2.0 * tf.math.square(std1)), axis=1) - 0.5 * self.d

    def mahalanobis(self, prob0, prob1):
      """
      https://en.wikipedia.org/wiki/Bhattacharyya_distance
      https://en.wikipedia.org/wiki/Mahalanobis_distance#Applications
      """
      mean0 = prob0[:, :self.d]
      std0 = prob0[:, self.d:]
      var0 = tf.square(std0)
      mean1 = prob1[:, :self.d]
      std1 = prob1[:, self.d:]
      var1 = tf.square(std1)
      mu = tf.expand_dims(mean0 - mean1, axis=1)
      # Assuming that each dimension is independent of each other.
      sigma = tf.linalg.inv(
          (tf.linalg.diag(var0) + tf.linalg.diag(var1)) / 2.0)
      return tf.squeeze(
          tf.sqrt(
              tf.maximum(
                  tf.matmul(
                      tf.matmul(mu, sigma),
                      tf.transpose(mu, perm=[0,2,1])
                  ), 1e-8,
              )
          ), axis=[1, 2]
      )

    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        return T.log(std_nd).sum(axis=1) + .5 * np.log(2 * np.pi * np.e) * self.d

    def sample(self, prob):
        mean_nd = prob[:, :self.d] 
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d).astype(floatX) * std_nd + mean_nd

    def maxprob(self, prob):
        return prob[:, :self.d]
