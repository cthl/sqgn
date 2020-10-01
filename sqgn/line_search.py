"""
line_search.py - Implementation of line search methods.

Author: Christopher Thiele
"""

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class LineSearch:
  """Abstract base class for line search methods."""
  def __init__(self, loss, weights, dws, grads, conf=None, sess=None):
    """
    Create a line search method.

    Arguments:
      loss:
        TF tensor that describes the loss function to be minimized.
      weights:
        List of TF variables w.r.t. which the loss is to be minimized.
      dws:
        Search direction (list of TF variables).
      grads:
        List of TF variables containing the *negative* gradient of the loss
        evaluated for the weights before the line search started.
      conf:
        Line search configuration.
      sess:
        TF session.
    """
    self._loss = loss
    self._weights = weights
    self._dws = dws
    self._grads = grads
    self._conf = conf

    if sess is None:
      self._sess = tf.get_default_session()
    else:
      self._sess = sess

    # Placeholder for the step size.
    self._step_size_placeh = tf.placeholder(shape=[], dtype=self._loss.dtype)

    # Inner product of gradient and search direction.
    # Required for the computation of sufficient decrease.
    self._grad_dot_dw = tf.zeros(shape=[], dtype=self._loss.dtype)
    for g, dw in zip(self._grads, self._dws):
      self._grad_dot_dw += tf.reduce_sum(-g*dw)

    self._update_weights_ops = self._update_weights()

  def _update_weights(self):
    ops = []

    for w, dw in zip(self._weights, self._dws):
      ops.append(tf.assign_add(w, self._step_size_placeh*dw))

    return tf.group(ops)

  def minimize(self, samples):
    """
    Perform line search.

    Arguments:
      samples:
        Feed dictionary to be used when sampling the loss function.
    """
    assert False, 'Cannot use abstract base class for line search!'

class FullStep(LineSearch):
  """Dummy line search method that simply performs a full step."""
  def __init__(self, loss, weights, dws, grads, conf=None, sess=None):
    """See 'LineSearch' class."""
    if conf is None:
      conf = FullStep.get_default_conf()

    super().__init__(loss, weights, dws, grads, conf, sess)

  @staticmethod
  def get_default_conf():
    """
    Obtain the default line search configuration.

    This particular line search method does not have any configuration
    parameters.
    """
    conf = {}

    return conf

  def minimize(self, samples):
    """See 'LineSearch' class."""
    fdict = {self._step_size_placeh: 1.0}
    self._sess.run(self._update_weights_ops, feed_dict=fdict)

class Armijo(LineSearch):
  """Classic Armijo line search method."""
  def __init__(self, loss, weights, dws, grads, conf=None, sess=None):
    """See 'LineSearch' class."""
    if conf is None:
      conf = Armijo.get_default_conf()

    super().__init__(loss, weights, dws, grads, conf, sess)

  @staticmethod
  def get_default_conf():
    """
    Obtain the default line search configuration.
    The configuration is a Python dictionary with the following entries:

      'alpha':
        Line search stops once the decrease in the loss function is sufficient,
        i.e., once it is at least alpha times the inner product of the gradient
        and the search direction (default: 1.0e-4).
      'beta':
        Factor by which the step size is reduced if the decrease in the loss
        function is insufficient (default: 0.5).
      'max_iter':
        Maximum number of step size reductions before the line search is deemed
        unsuccessful (default: 25).
    """
    conf = {}
    conf['alpha'] = 1.0e-4
    conf['beta'] = 0.5
    conf['max_iter'] = 25

    return conf

  def minimize(self, samples):
    """See 'LineSearch' class."""
    beta = self._conf['beta']

    # Compute initial loss.
    initial_loss = self._loss.eval(session=self._sess, feed_dict=samples)

    # Compute sufficient decrease.
    sufficient_decrease = self._conf['alpha'] \
                          *self._grad_dot_dw.eval(session=self._sess)

    k = 0
    while k < self._conf['max_iter']:
      # Apply the update to the weights.
      if k == 0:
        # Take the full step in the first iteration.
        step_size = 1.0
      else:
        # Keep in mind that beta^(k-1)*dw has already been added to the weights
        # in the previous iteration.
        # Hence, we only need to backtrack using a *negative* step size.
        step_size = beta**(k - 1)*(beta - 1.0)
      fdict = {self._step_size_placeh: step_size}
      self._sess.run(self._update_weights_ops, feed_dict=fdict)

      # Compute decrease.
      loss = self._loss.eval(session=self._sess, feed_dict=samples)
      decrease = loss - initial_loss

      print('Armijo: k = %d, lambda = %.2e, decr. = %.2e, suff. decr. = %.2e' % (k, step_size, decrease, sufficient_decrease))

      if decrease < sufficient_decrease:
        return

      k += 1

    print('Line search was unsuccessful! (decr. = %.2e)' % decrease)
