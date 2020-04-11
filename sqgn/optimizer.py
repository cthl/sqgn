"""
optimizer.py - Abstract base class for all optimizers.

Author: Christopher Thiele
"""

import tensorflow as tf

class Optimizer:
  """Abstract optimizer base class."""
  def __init__(self, loss, weights, conf=None, sess=None):
    """
    Arguments:
      loss:
        Loss function to be minimized (TF tensor).
      weights:
        List of trainable TF variables that affect the loss function.
      conf:
        Optimizer configuration.
        If 'None', the default configuration is used.
      sess:
        TF session.
        If 'None', the default TF session is used if available.
    """
    self._loss = loss
    self._weights = weights

    # The abstract base class does not have a default configuration.
    # If 'conf' is 'None', the child class will create a default configuration.
    self._conf = conf

    if sess is None:
      self._sess = tf.get_default_session()
    else:
      self._sess = sess

  @property
  def loss(self):
    """Loss function."""
    return self._loss
  
  @property
  def weights(self):
    """Weights."""
    return self._weights

  @property
  def conf(self):
    """Optimizer configuration."""
    return self._conf

  @property
  def sess(self):
    """TF session."""
    return self._sess

  def minimize(self, samples, samples_hessian=None):
    """
    Perform a single iteration of the optimization algorithm.

    Arguments:
      samples:
        Feed dictionary to be used when sampling the loss function and its
        gradient during optimization.
      samples_hessian:
        Optional feed dictionary to be used when sampling the Hessian of the
        loss function during optimization.
        If 'None', the Hessian is sampled with the same data that is used for
        the loss function and its gradient.
    """
    assert False, 'Cannot minimize with abstract optimizer base class!'
