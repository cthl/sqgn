"""
gradient_aggregator.py - Implementation of variance reduction methods.

Author: Christopher Thiele
"""

import tensorflow as tf

class GradientAggregator:
  """Abstract gradient aggregator base class."""
  def __init__(self, loss, weights, sess=None):
    """
    Create a gradient aggregator.

    See 'Optimizer' class for a description of the arguments.
    """
    self._loss = loss
    self._weights = weights

    if sess is None:
      self._sess = tf.get_default_session()
    else:
      self._sess = sess

    dtype = self._loss.dtype
    shapes = [w.shape.as_list() for w in self._weights]

    self._grads = tf.gradients(loss, weights)

    # Create variables to store the reference gradient and the weights with
    # which the reference gradient was evaluated.
    self._ws_ref = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]
    self._grads_ref = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) \
                       for s in shapes]

    # Since we need to evaluate the gradient with different weights, we need
    # backup variables to swap weights.
    self._ws_temp = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) \
                     for s in shapes]

    # Finally, we need variables to store the actual gradient.
    self._grads_aggregated = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) \
                              for s in shapes]

    # Variables to keep track of the number of samples used to compute the
    # reference gradient.
    self._num_ref_samples = tf.Variable(tf.zeros(shape=[], dtype=tf.int32))
    self._num_ref_samples_batch = tf.placeholder(shape=[], dtype=tf.int32)

    self._update_ref_ops = self._update_ref()
    self._reset_ref_ops = self._reset_ref()
    self._ref_add_batch_ops = self._ref_add_batch()
    self._finalize_ref_ops = self._finalize_ref()

  def _update_ref(self):
    ops = []

    for w_ref, w, g_ref, g \
    in zip(self._ws_ref, self._weights, self._grads_ref, self._grads):
      ops.append(tf.assign(w_ref, w))
      ops.append(tf.assign(g_ref, g))

    return tf.group(ops)

  def _reset_ref(self):
    ops = []

    for w_ref, w, g_ref in zip(self._ws_ref, self._weights, self._grads_ref):
      ops.append(tf.assign(w_ref, w))
      ops.append(tf.assign(g_ref, tf.zeros_like(g_ref)))
      ops.append(tf.assign(self._num_ref_samples, 0))

    return tf.group(ops)

  def _ref_add_batch(self):
    ops = []

    for g_ref, g in zip(self._grads_ref, self._grads):
      ops.append(tf.assign_add(g_ref,
                               g*tf.cast(self._num_ref_samples_batch,
                                         dtype=self._loss.dtype)))
    ops.append(tf.assign_add(self._num_ref_samples,
                             self._num_ref_samples_batch))

    return tf.group(ops)

  def _finalize_ref(self):
    ops = []

    for g_ref in self._grads_ref:
      ops.append(tf.assign(g_ref,
                           g_ref/tf.cast(self._num_ref_samples,
                                         dtype=self._loss.dtype)))

    return tf.group(ops)

  def update_reference(self, samples):
    """Update the reference gradient using a single batch."""
    self._sess.run(self._update_ref_ops, feed_dict=samples)

  def reset_reference(self):
    """Reset the reference gradient."""
    self._sess.run(self._reset_ref_ops)

  def reference_add(self, samples, num_samples):
    """
    Add sampled gradient to the reference gradient.
    Call this method repeatedly to construct the reference gradient from
    multiple batches, and call finalize_reference() afterwards.
    """

    # Add the number of samples to the feed dictionary.
    samples[self._num_ref_samples_batch] = num_samples
    self._sess.run(self._ref_add_batch_ops, feed_dict=samples)

  def finalize_reference(self):
    """Finish the reference gradient update."""
    self._sess.run(self._finalize_ref_ops)

  def compute_grad(self, samples):
    """
    Compute the current gradient using the reference gradient and the given
    samples.
    """
    assert False, 'Cannot compute aggregated gradient with abstract base class!'

  def get_grad(self):
    """
    Get a list of TF variables that store the gradients after 'compute_grad' has
    been called.
    """
    return self._grads_aggregated

class RawGradient(GradientAggregator):
  def __init__(self, loss, weights, sess=None):
    """See 'GradientAggregator' class."""
    super().__init__(loss, weights, sess)

    self._copy_grad_ops = self._copy_grad()

  def _copy_grad(self):
    ops = []

    for g_agg, g in zip(self._grads_aggregated, self._grads):
      ops.append(tf.assign(g_agg, g))

    return tf.group(ops)

  def update_reference(self, samples):
    """See 'GradientAggregator' class."""

    # This class does not use the reference gradient, so we can save the cost of
    # storing it in the first place.
    pass

  def compute_grad(self, samples):
    """See 'GradientAggregator' class."""
    self._sess.run(self._copy_grad_ops, feed_dict=samples)

class SVRG(GradientAggregator):
  """
  This class implements gradient aggregation as it is used in the stochastic
  variance reduced gradient (SVRG) descent method.
  """
  def __init__(self, loss, weights, sess=None):
    """See 'GradientAggregator' class."""
    super().__init__(loss, weights, sess)

    self._eval_grad_ops = self._eval_grad()
    self._add_ref_ops = self._add_ref()
    self._sub_mixed_grad_ops = self._sub_mixed_grad()
    self._restore_weights_ops = self._restore_weights()

  def _eval_grad(self):
    ops = []

    # Evaluate the current gradient and save the current weights.
    for g_agg, g, w_temp, w \
    in zip(self._grads_aggregated, self._grads, self._ws_temp, self._weights):
      ops.append(tf.assign(g_agg, g))
      ops.append(tf.assign(w_temp, w))

    return tf.group(ops)

  def _add_ref(self):
    ops = []

    # Add the reference gradient and copy the reference weights.
    for g_agg, g_ref, w, w_ref in zip(self._grads_aggregated,
                                      self._grads_ref,
                                      self._weights,
                                      self._ws_ref):
      ops.append(tf.assign_add(g_agg, g_ref))
      ops.append(tf.assign(w, w_ref))

    return tf.group(ops)

  def _sub_mixed_grad(self):
    ops = []

    # Subtract the current gradient evaluated with the reference weights.
    for g_agg, g in zip(self._grads_aggregated, self._grads):
      ops.append(tf.assign_sub(g_agg, g))

    return tf.group(ops)

  def _restore_weights(self):
    ops = []

    for w, w_temp in zip(self._weights, self._ws_temp):
      ops.append(tf.assign(w, w_temp))

    return tf.group(ops)

  def compute_grad(self, samples):
    """See 'GradientAggregator' class."""
    self._sess.run(self._eval_grad_ops, feed_dict=samples)
    self._sess.run(self._add_ref_ops)
    self._sess.run(self._sub_mixed_grad_ops, feed_dict=samples)
    self._sess.run(self._restore_weights_ops)
