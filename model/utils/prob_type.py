import theano
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import scipy.stats

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

    def py_likelihood(self, a, prob):
      # Must be one dimensional.
      mean0 = prob[:self.d]
      std0 = prob[self.d:]
      return scipy.stats.multivariate_normal(mean0, std0).pdf(a)

    def kl(self, prob0, prob1):
      mean0 = prob0[:, :self.d]
      std0 = prob0[:, self.d:]
      var0 = tf.square(std0)
      mean1 = prob1[:, :self.d]
      std1 = prob1[:, self.d:]
      var1 = tf.square(std1)

      def _one_dim_kl(mu1, sigma1, stddev1, mu2, sigma2, stddev2):
        # This func is not essential though. Because the one-dimensional case
        # is covered by the following k-dimensional case, where k can be 1.
        t1 = tf.math.log(stddev2 / tf.maximum(stddev1, 1e-8))
        t2 = (sigma1 + tf.math.square(mu1 - mu2)) / (
            2.0 * tf.maximum(sigma2, 1e-8))
        return t1 + t2 - 1.0 / 2.0

      if self.d == 1:
        return _one_dim_kl(mean0, var0, std0, mean1, var1, std1)

      return tf.reduce_sum(tf.math.log(std1 / std0), axis=1) + tf.reduce_sum((tf.math.square(std0) + tf.math.square(mean0 - mean1)) / (2.0 * tf.math.square(std1)), axis=1) - 0.5 * self.d

    def tv(self, prob0, prob1):
      """
      In general, there is no known closed form for the total variation distance
      between two multivarate normal distribution, but the following references
      provide us a lower bound and upper bound for that.

      https://arxiv.org/pdf/1810.08693.pdf
      https://en.wikipedia.org/wiki/Total_variation
      """
      mean0 = prob0[:, :self.d]
      std0 = prob0[:, self.d:]
      var0 = tf.square(std0)
      mean1 = prob1[:, :self.d]
      std1 = prob1[:, self.d:]
      var1 = tf.square(std1)

      def _one_dim_tv(mu1, sigma1, stddev1, mu2, sigma2, stddev2):
        # This func is essential, as it doesn't involve the calculation of
        # sum of squared eigenvalues.
        t1 = tf.abs(sigma1 - sigma2) / tf.maximum(sigma1, 1e-8)
        t2 = tf.abs(mu1 - mu2) / tf.maximum(stddev1, 1e-8)
        lower = 1.0 / 200.0 * tf.minimum(1.0, tf.maximum(t1, 40.0 * t2))
        upper = 3.0 / 2.0 * t1 + 1.0 / 2.0 * t2
        return lower, upper

      if self.d == 1:
        lower, upper = _one_dim_tv(mean0, var0, std0, mean1, var1, std1)
        return upper

      def _tv(mu1, sigma1, mu2, sigma2):
        v = tf.expand_dims(mu1 - mu2, axis=1)
        t1 = tf.abs(tf.matmul(
            tf.matmul(v, sigma1 - sigma2), tf.transpose(v, perm=[0,2,1])
        )) / tf.maximum(tf.matmul(
            tf.matmul(v, sigma1), tf.transpose(v, perm=[0,2,1])
        ), 1e-8)
        t2 = tf.matmul(v, tf.transpose(v, perm=[0,2,1])) / tf.sqrt(
            tf.maximum(
                tf.matmul(
                    tf.matmul(v, sigma1),
                    tf.transpose(v, perm=[0,2,1])
                ), 1e-8,
            )
        )
        t3 = tf.sqrt(
            tf.maximum(
                tf.linalg.trace(tf.square(
                    tf.matmul(
                        tf.linalg.inv(sigma1), sigma2
                    ) - tf.eye(self.d, batch_shape=[tf.shape(mu1)[0]])
                )), 1e-8
            )
        )
        return tf.maximum(tf.maximum(t1, t2), t3)

      tv = _tv(mean0, tf.linalg.diag(var0), mean1, tf.linalg.diag(var1))
      dom = tf.minimum(1.0, tv)
      lower = 1.0 / 200.0 * dom
      upper = 9.0 / 2.0 * dom
      return upper

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
      sigma = tf.linalg.inv(tf.linalg.diag((var0 + var1) / 2.0))
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

    def wasserstein(self, prob0, prob1):
      """
      https://en.wikipedia.org/wiki/Wasserstein_metric
      """
      mean0 = prob0[:, :self.d]
      std0 = prob0[:, self.d:]
      var0 = tf.square(std0)
      mean1 = prob1[:, :self.d]
      std1 = prob1[:, self.d:]
      var1 = tf.square(std1)
      n = tf.square(tf.linalg.norm(mean0 - mean1, ord=2, axis=1))
      sqrtc2 = tf.sqrt(var1)
      c = var0 + var1 - 2 * tf.sqrt(
          tf.matmul(tf.matmul(sqrtc2, var0), sqrtc2))
      return n + tf.linalg.trace(c)

    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        return T.log(std_nd).sum(axis=1) + .5 * np.log(2 * np.pi * np.e) * self.d

    def sample(self, prob):
        mean_nd = prob[:, :self.d] 
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d).astype(floatX) * std_nd + mean_nd

    def maxprob(self, prob):
        return prob[:, :self.d]
