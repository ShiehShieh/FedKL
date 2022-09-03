import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class DecayBasedGradientDescent(optimizer.Optimizer):
  """Implementation of Decay-Based Gradient Descent, i.e., FMARL optimizer."""
  def __init__(self, learning_rate=0.01, lamb=0.98, use_locking=False, name="DBPG"):
    super(DecayBasedGradientDescent, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._lamb = lamb

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._lamb_t = None

  def _df(self, lamb, cnt):
    return math_ops.pow(lamb, math_ops.div(cnt, 2.0))

  def _prepare(self):
    self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
    self._lamb_t = ops.convert_to_tensor(self._lamb, name="fmarl_lambda")

  def _create_slots(self, var_list):
    # Create slots for the global solution.
    # for v in var_list:
    #     self._zeros_slot(v, "cnt", self._name)
    self.lamb = tf.Variable(initial_value=self._lamb)
    self._zeros_slot(self.lamb, "cnt", self._name)
    self.var_list = var_list

  def _apply_dense(self, grad, var):
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    lamb_t = math_ops.cast(self._lamb_t, var.dtype.base_dtype)
    cnt = self.get_slot(self.lamb, "cnt")

    cnt_update = state_ops.assign_add(cnt, 1.0)
    with ops.control_dependencies([cnt_update]):
      var_update = state_ops.assign_sub(var, lr_t * self._df(lamb_t, cnt) * grad)

    return control_flow_ops.group(*[var_update,])

  def _resource_apply_dense(self, grad, handle):
    """Add ops to apply dense gradients to the variable `handle`.
    Args:
      grad: a `Tensor` representing the gradient.
      handle: a `Tensor` of dtype `resource` which points to the variable
       to be updated.
    Returns:
      An `Operation` which updates the value of the variable.
    """
    return self._apply_dense(grad, handle)
  
  def _apply_sparse_shared(self, grad, var, indices, scatter_add):

    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    lamb_t = math_ops.cast(self._lamb_t, var.dtype.base_dtype)
    cnt = self.get_slot(self.lamb, "cnt")

    cnt_update = state_ops.assign_add(cnt, 1.0)
    with ops.control_dependencies([cnt_update]):
      var_update = state_ops.assign_sub(var, lr_t * self._df(lamb_t, cnt) * grad)

    return control_flow_ops.group(*[var_update,])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(x, i, v))

  def reset_params(self, sess):
    cnt = self.get_slot(self.lamb, "cnt")
    cnt.load(0.0, sess)

  def get_var_list(self):
    return self.var_list

  def get_lambda(self):
    return self.get_slot(self.lamb, "cnt")
