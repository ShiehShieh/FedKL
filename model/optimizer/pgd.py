# Copyright to https://github.com/litian96/FedProx/blob/master/flearn/optimizer/pgd.py.

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class PerturbedGradientDescent(optimizer.Optimizer):
    """Implementation of Perturbed Gradient Descent, i.e., FedProx optimizer"""
    def __init__(self, learning_rate=0.001, mu=0.01, use_locking=False, name="PGD"):
        super(PerturbedGradientDescent, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._mu = mu
       
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._mu_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._mu, name="prox_mu")

    def _create_slots(self, var_list):
        # Create slots for the global solution.
        for v in var_list:
            self._zeros_slot(v, "vstar", self._name)
        self.var_list = var_list

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        var_update = state_ops.assign_sub(var, lr_t*(grad + mu_t*(var-vstar)))

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
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        v_diff = state_ops.assign(vstar, mu_t * (var - vstar), use_locking=self._use_locking)

        with ops.control_dependencies([v_diff]):  # run v_diff operation before scatter_add
            scaled_grad = scatter_add(vstar, indices, grad)
        var_update = state_ops.assign_sub(var, lr_t * scaled_grad)

        return control_flow_ops.group(*[var_update,])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(x, i, v))
    

    def set_params(self, sess, cog):
      for variable, value in zip(self.get_var_list(), cog):
        vstar = self.get_slot(variable, "vstar")
        vstar.load(value, sess)

    def get_var_list(self):
      return self.var_list
