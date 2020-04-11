"""
rmsprop.py - Root mean square propagation optimizer.

Author: Christopher Thiele
"""

from sqgn.optimizer import Optimizer

import tensorflow as tf

class RMSprop(Optimizer):
  def __init__(self, loss, weights, conf=None, sess=None):
    """See 'Optimizer' class."""
    if conf is None:
      conf = RMSprop.get_default_conf()

    super().__init__(loss, weights, conf, sess)

    # Create variables for running averages.
    shapes = [w.shape.as_list() for w in self._weights]
    vs = [tf.Variable(tf.zeros(shape=s, dtype=loss.dtype)) for s in shapes]

    grads = tf.gradients(self._loss, self._weights)

    self._update_running_avg_ops = self._update_running_avg(vs, grads)
    self._update_weights_ops = self._update_weights(vs, grads)

  def _update_running_avg(self, vs, grads):
    ops = []

    gamma = self._conf['gamma']
    for v, g in zip(vs, grads):
      # Compute running average.
      v_new = gamma*v + (1.0 - gamma)*tf.square(g)
      ops.append(tf.assign(v, v_new))

    return tf.group(ops)

  def _update_weights(self, vs, grads):
    ops = []

    for w, g, v in zip(self._weights, grads, vs):
      w_new = w - self._conf['lr']/(tf.sqrt(v) + self._conf['eps'])*g
      ops.append(tf.assign(w, w_new))

    return tf.group(ops)

  @staticmethod
  def get_default_conf():
    """
    Obtain the default optimizer configuration.
    The configuration is a Python dictionary with the following entries:

      'lr':
        Learning rate (default: 1.0e-3).
      'eps':
        Small positive constant to prevent division by zero
        (default: 1.0e-7).
      'gamma':
        Forgetting factor (default: 0.9).
    """
    conf = {}
    conf['lr'] = 1.0e-3
    conf['eps'] = 1.0e-7
    conf['gamma'] = 0.9

    return conf

  def minimize(self, samples, samples_hessian=None):
    """See 'Optimizer' class."""
    self._sess.run(self._update_running_avg_ops, feed_dict=samples)
    self._sess.run(self._update_weights_ops, feed_dict=samples)
