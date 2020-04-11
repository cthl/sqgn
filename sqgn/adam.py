"""
adam.py - Adaptive moment estimation method.

Author: Christopher Thiele
"""

from sqgn.optimizer import Optimizer

import tensorflow as tf

class Adam(Optimizer):
  """
  Adaptive moment estimation optimizer.
  """
  def __init__(self, loss, weights, conf=None, sess=None):
    """See 'Optimizer' class."""
    if conf is None:
      conf = Adam.get_default_conf()

    super().__init__(loss, weights, conf, sess)

    # Create variables for the momentum and velocity terms.
    k = tf.Variable(0, dtype=tf.int32)
    shapes = [w.shape.as_list() for w in self._weights]
    ms = [tf.Variable(tf.zeros(shape=s, dtype=loss.dtype)) for s in shapes]
    vs = [tf.Variable(tf.zeros(shape=s, dtype=loss.dtype)) for s in shapes]

    self._update_momentum_and_velocity_ops \
      = self._update_momentum_and_velocity(ms, vs)
    self._update_weights_ops = self._update_weights(k, ms, vs)
    self._increment_iter_ops = self._increment_iter(k)

  def _update_momentum_and_velocity(self, ms, vs):
    ops = []

    beta1 = self._conf['beta1']
    beta2 = self._conf['beta2']

    grads = tf.gradients(self._loss, self._weights)

    for m, v, g in zip(ms, vs, grads):
      m_new = beta1*m + (1.0 - beta1)*g
      v_new = beta2*v + (1.0 - beta2)*tf.square(g)
      ops.append(tf.assign(m, m_new))
      ops.append(tf.assign(v, v_new))

    return tf.group(ops)

  def _update_weights(self, k, ms, vs):
    ops = []

    k_new = k + 1

    # Coefficients beta1 and beta2 and their exponent
    beta1 = self._conf['beta1']
    beta2 = self._conf['beta2']
    exp = tf.cast(k_new, dtype=self._loss.dtype)

    for w, m, v in zip(self._weights, ms, vs):
      coeff = (1.0 - tf.pow(beta1, exp)) \
              *(tf.sqrt(v)/tf.sqrt(1.0 - tf.pow(beta2, exp))) \
              + self._conf['eps']
      w_new = w - self._conf['lr']/coeff*m

      ops.append(tf.assign(w, w_new))

    return tf.group(ops)

  def _increment_iter(self, k):
    return tf.assign_add(k, 1)

  @staticmethod
  def get_default_conf():
    """
    Obtain the default optimizer configuration.
    The configuration is a Python dictionary with the following entries:

      'lr':
        Learning rate (default: 1.0e-3).
      'beta1':
        Coefficient smaller than, but close to, one (default: 0.9).
      'beta2':
        Coefficient smaller than, but close to, one (default: 0.999).
      'eps':
        Small positive constant that prevents division by zero
        (default: 1.0e-7).
    """
    conf = {}
    conf['lr'] = 1.0e-3
    conf['beta1'] = 0.9
    conf['beta2'] = 0.999
    conf['eps'] = 1.0e-7

    return conf

  def minimize(self, samples, samples_hessian=None):
    """See 'Optimizer' class."""
    self._sess.run(self._update_momentum_and_velocity_ops, feed_dict=samples)
    self._sess.run(self._update_weights_ops, feed_dict=samples)
    self._sess.run(self._increment_iter_ops)
