"""
sgd.py - Stochastic gradient descent method.

Author: Christopher Thiele
"""

import sqgn
from sqgn.optimizer import Optimizer

import tensorflow as tf

class SGD(Optimizer):
  """
  Stochastic gradient descent optimizer.
  """
  def __init__(self, loss, weights, conf=None, sess=None):
    """See 'Optimizer' class."""
    if conf is None:
      conf = SGD.get_default_conf()

    super().__init__(loss, weights, conf, sess)

    # Create the gradient aggregator.
    self._grad_agg = sqgn.create_gradient_aggregator(self._conf['grad_agg'],
                                                     self._loss,
                                                     self._weights,
                                                     self._sess)
    grads = self._grad_agg.get_grad()

    self._sgd_update_ops = self._sgd_update(grads)

  def _sgd_update(self, grads):
    ops = []
    
    for w, g in zip(self._weights, grads):
      w_new = w - self._conf['lr']*g
      ops.append(tf.assign(w, w_new))
    
    return tf.group(ops)

  def get_gradient_aggregator(self):
    return self._grad_agg

  @staticmethod
  def get_default_conf():
    """
    Obtain the default optimizer configuration.
    The configuration is a Python dictionary with the following entries:

      'lr':
        Learning rate (default: 1.0e-2).
      'grad_agg':
        Gradient aggregation method (default: 'raw_grad').
    """
    conf = {}
    conf['lr'] = 1.0e-2
    conf['grad_agg'] = 'raw_grad'

    return conf

  def minimize(self, samples, samples_hessian=None):
    """See 'Optimizer' class."""
    self._grad_agg.compute_grad(samples)
    self._sess.run(self._sgd_update_ops, feed_dict=samples)
