"""
lbfgs.py - Online BFGS method.

Author: Christopher Thiele
"""

import sqgn
from sqgn.optimizer import Optimizer

import tensorflow as tf

class LBFGS(Optimizer):
  """
  Online, i.e. stochastic, limited-memory Broyden–Fletcher–Goldfarb–Shanno
  (LBFGS) optimizer.

  This implementation is based on a 2007 paper by Schraudolph et al.
  (title: A Stochastic Quasi-Newton Method for Online Convex Optimization).

  Note that without further modifications, LBFGS cannot handle negative
  curvature and the resulting indefinite approximations to the Hessians.
  If the Hessian approximations become indefinite, the method will fail, often
  producing NaNs.
  Increasing the regularization parameter can help to avoid indefinite Hessian
  approximations.
  """
  def __init__(self, loss, weights, conf=None, sess=None):
    """See 'Optimizer' class."""
    if conf is None:
      conf = LBFGS.get_default_conf()

    super().__init__(loss, weights, conf, sess)

    # Compute gradients.
    grads = tf.gradients(self._loss, self._weights)

    # Create TensorFlow variables to store the histories of the updates s and
    # the gradient differences y.
    dtype = self._loss.dtype
    shapes = [w.shape.as_list() for w in self._weights]
    hist_shapes = [[self._conf['hist']] + s for s in shapes]
    hist_idx = tf.Variable(tf.zeros(shape=[], dtype=tf.int32))
    hist_size = tf.Variable(tf.zeros(shape=[], dtype=tf.int32))
    s_hists = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in hist_shapes]
    y_hists = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in hist_shapes]
    sTy_val = tf.Variable(tf.zeros(shape=[], dtype=dtype))
    # Create variables to store the gradients before the weight updates.
    # These gradients are needed to compute the gradient difference y_k later.
    grads_prev = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]

    self._save_grads_ops = self._save_grads(grads, grads_prev)
    self._compute_step_ops = self._compute_step(hist_idx,
                                                hist_size,
                                                grads,
                                                s_hists,
                                                y_hists)
    self._update_weights_ops = self._update_weights(hist_idx, s_hists)
    self._compute_y_ops = self._compute_y(hist_idx,
                                          grads,
                                          grads_prev,
                                          s_hists,
                                          y_hists)
    self._compute_sTy_ops = self._compute_sTy(hist_idx,
                                              s_hists,
                                              y_hists,
                                              sTy_val)
    self._update_history_ops = self._update_history(hist_idx,
                                                    hist_size,
                                                    sTy_val)

  def _save_grads(self, grads, grads_prev):
    ops = []

    for g, g_prev in zip(grads, grads_prev):
      ops.append(tf.assign(g_prev, g))

    return tf.group(ops)

  def _compute_step(self, hist_idx, hist_size, grads, s_hists, y_hists):
    ops = []

    # Create tensors to compute dot products from the histories of s and y.
    sTy_hist = []
    yTy_hist = []
    # We construct the list so that sTy_history[-i] will compute the dot product
    # (s_{k-i}, y_{k-i}) etc.
    for i in reversed(range(1, self._conf['hist'] + 1)):
      sTy = tf.zeros([])
      yTy = tf.zeros([])
      idx = tf.mod(hist_idx - i, self._conf['hist'])
      for s_hist, y_hist in zip(s_hists, y_hists):
        sTy += tf.reduce_sum(s_hist[idx]*y_hist[idx])
        yTy += tf.reduce_sum(y_hist[idx]*y_hist[idx])
      sTy_hist.append(sTy)
      yTy_hist.append(yTy)

    # Start with the negative gradient.
    ps = []
    for g in grads:
      ps.append(-g)

    # First stage of the update (alg. 3, step 2 in the paper)
    alphas = []

    # Create a TensorFlow group that bundles all updates from an iteration
    # within the first stage.
    for i in range(1, self._conf['hist'] + 1):
      idx = tf.mod(hist_idx - i, self._conf['hist'])
      # Compute coefficient alpha (alg. 3, step 2a).
      sTp = tf.zeros(shape=[])
      for p, s_hist in zip(ps, s_hists):
        sTp += tf.reduce_sum(s_hist[idx]*p)
      alpha = tf.cond(i <= hist_size,
                      lambda: sTp/sTy_hist[-i],
                      lambda: tf.zeros(shape=[]))
      alphas.append(alpha)
      # Update direction (alg. 3, step 2b).
      for j, y_hist in enumerate(y_hists):
        ps[j] -= alpha*y_hist[idx]

    # Second stage of the update (alg. 3, step 3 and eq. 14)
    coeff = tf.zeros(shape=[])
    for i in range(1, self._conf['hist'] + 1):
      coeff = tf.cond(i <= hist_size,
                      lambda: coeff + sTy_hist[-i]/yTy_hist[-i],
                      lambda: coeff)
    coeff = tf.cond(tf.equal(hist_size, 0),
                    lambda: self._conf['eps'],
                    lambda: coeff/tf.cast(hist_size, dtype=coeff.dtype))
    #coeff = tf.Print(coeff, [hist_size], message='hist_size = ')
    for j in range(len(ps)):
      ps[j] *= coeff

    # Third stage of the update (alg. 3, step 4)
    for i in reversed(range(1, self._conf['hist'] + 1)):
      idx = tf.mod(hist_idx - i, self._conf['hist'])
      yTp = tf.zeros(shape=[])
      for p, y_hist in zip(ps, y_hists):
        yTp += tf.reduce_sum(y_hist[idx]*p)
      beta = yTp/sTy_hist[-i]
      alpha_minus_beta = tf.cond(i <= hist_size,
                                 lambda: alphas[i - 1] - beta,
                                 lambda: tf.zeros(shape=[]))
      for j, s_hist in enumerate(s_hists):
        ps[j] += alpha_minus_beta*s_hist[idx]

    # Save update.
    for p, s_hist in zip(ps, s_hists):
      s = self._conf['lr']*p
      ops.append(tf.scatter_update(s_hist, hist_idx, s))

    return tf.group(ops)

  def _update_weights(self, hist_idx, s_hists):
    ops = []

    # Update weights.
    for w, s_hist in zip(self._weights, s_hists):
      w_new = w + s_hist[hist_idx]
      ops.append(tf.assign(w, w_new))

    return tf.group(ops)

  def _compute_y(self, hist_idx, grads, grads_prev, s_hists, y_hists):
    ops = []

    for s_hist, y_hist, g, g_prev in zip(s_hists, y_hists, grads, grads_prev):
      # Compute the difference of gradients and add the regularization term.
      y_new = g - g_prev + self._conf['reg']*s_hist[hist_idx]
      ops.append(tf.scatter_update(y_hist, hist_idx, y_new))

    return tf.group(ops)

  def _compute_sTy(self, hist_idx, s_hists, y_hists, sTy_val):
    sTy = tf.zeros(shape=[], dtype=self._loss.dtype)
    for s_hist, y_hist in zip(s_hists, y_hists):
      sTy += tf.reduce_sum(s_hist[hist_idx]*y_hist[hist_idx])

    return tf.assign(sTy_val, sTy)

  def _update_history(self, hist_idx, hist_size, sTy_val):
    ops = []

    neg_curv_update = self._conf['neg_curv_update']

    if neg_curv_update == 'ignore':
      # Perform standard update.
      hist_idx_new = hist_idx + 1
      hist_size_new = hist_size + 1
    elif neg_curv_update == 'reset':
      # Reset the entire approximation of the Hessian inverse if negative
      # curvature is encountered.
      hist_idx_new = tf.cond(sTy_val > 0.0,
                             lambda: hist_idx + 1,
                             lambda: tf.zeros(shape=[], dtype=hist_idx.dtype))
      hist_size_new = tf.cond(sTy_val > 0.0,
                              lambda: hist_size + 1,
                              lambda: tf.zeros(shape=[], dtype=hist_size.dtype))
    elif neg_curv_update == 'discard':
      # Ignore the update if negative curvature is encountered.
      hist_idx_new = tf.cond(sTy_val > 0.0,
                             lambda: hist_idx + 1,
                             lambda: hist_idx)
      hist_size_new = tf.cond(sTy_val > 0.0,
                              lambda: hist_size + 1,
                              lambda: hist_size)
    else:
      assert False, 'User must specify how to treat negative curvature!'

    # Make sure that the history index and the history size do not leave their
    # bounds.
    hist_idx_new = tf.mod(hist_idx_new, self._conf['hist'])
    hist_size_new = tf.minimum(hist_size_new, self._conf['hist'])

    ops.append(tf.assign(hist_idx, hist_idx_new))
    ops.append(tf.assign(hist_size, hist_size_new))

    return tf.group(ops)

  @staticmethod
  def get_default_conf():
    """
    Obtain the default optimizer configuration.
    The configuration is a Python dictionary with the following entries:

      'lr':
        Learning rate (called 'eta' in the paper, default: 1.0).
      'eps':
        Small positive constant to ensure that the update in the first iteration
        is sufficiently small (called 'epsilon' in the paper, default: 1.0e-4).
      'hist':
        Number of previous iterations to consider in the LBFGS update
        (called 'm' in the paper, default: 5).
      'reg':
        Non-negative regularization parameter (called 'lambda' in the paper,
        default: 0.0).
      'neg_curv_update':
        Determines how the Hessian inverse approximation should be updated if
        negative curvature is encountered.
        If set to 'ignore', the standard update is performed regardless of the
        negative curvature.
        If set to 'discard', the Hessian inverse approximation remains
        unchanged.
        If set to 'reset', the Hessian inverse approximation is set to the
        identity, effectively restarting the method with a step of gradient
        descent.
        (default: 'ignore')
    """
    conf = {}
    conf['lr'] = 1.0
    conf['eps'] = 1.0e-4
    conf['hist'] = 5
    conf['reg'] = 0.0
    conf['neg_curv_update'] = 'ignore'

    return conf

  def minimize(self, samples, samples_hessian=None):
    """See 'Optimizer' class."""
    self._sess.run(self._save_grads_ops, feed_dict=samples)
    self._sess.run(self._compute_step_ops, feed_dict=samples)
    self._sess.run(self._update_weights_ops)
    self._sess.run(self._compute_y_ops, feed_dict=samples)
    self._sess.run(self._compute_sTy_ops)
    self._sess.run(self._update_history_ops)

class SQN(Optimizer):
  """
  Stochastic quasi-Newton (SQN) method.

  This method improves the online LBFGS method by using Hessian-vector products
  instead of gradient differences to construct the approximation the the
  Hessian inverse.
  This makes the iteration less sensitive to noisy gradients.
  It also allows us to apply a BFGS-type method for non-convex losses, since we
  can replace the Hessian by the Gauss-Newton approximation in order to ensure
  the positive definiteness of the Hessian inverse approximation.
  Finally, since the method does not rely on line search or trust regions, we
  can include variance reduction techniques for the gradient evaluations.
  """
  def __init__(self, loss, weights, conf=None, sess=None):
    """See 'Optimizer' class."""
    if conf is None:
      conf = SQN.get_default_conf()
    if conf['gauss_newton'] and conf['num_samples'] is None:
      assert False, 'User must provide number of samples for Gauss-Newton!'
    if conf['gauss_newton'] and conf['pred'] is None:
      assert False, 'User must provide a residual function for Gauss-Newton!'

    super().__init__(loss, weights, conf, sess)

    self._iter = 0

    # Create the gradient aggregator.
    self._grad_agg = sqgn.create_gradient_aggregator(self._conf['grad_agg'],
                                                     self._loss,
                                                     self._weights,
                                                     self._sess)
    grads = self._grad_agg.get_grad()

    # Create TensorFlow variables to store the histories of s and y.
    dtype = self._loss.dtype
    shapes = [w.shape.as_list() for w in self._weights]
    hist_shapes = [[self._conf['hist']] + s for s in shapes]
    hist_idx = tf.Variable(tf.zeros(shape=[], dtype=tf.int32))
    hist_size = tf.Variable(tf.zeros(shape=[], dtype=tf.int32))
    s_hists = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in hist_shapes]
    y_hists = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in hist_shapes]

    # Create a tensor that represents the action of either the Hessian or the
    # Gauss-Newton approximation.
    zs = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]
    if self._conf['gauss_newton']:
      Hzs = sqgn.gauss_newton.JTHJv(self._conf['pred'],
                                    self._loss,
                                    self._conf['num_samples'],
                                    self._weights,
                                    zs)
    else:
      # Construct the Hessian-vector product.
      grad_dot_z = tf.zeros(shape=[], dtype=dtype)
      for g, z in zip(grads, zs):
        grad_dot_z += tf.reduce_sum(g*z)
      Hzs = tf.gradients(grad_dot_z, self._weights)

    # Add regularization.
    i = 0
    for z, Hz in zip(zs, Hzs):
      Hzs[i] = Hz + self._conf['reg']*z
      i += 1

    self._compute_step_ops = self._compute_step(hist_idx,
                                                hist_size,
                                                grads,
                                                s_hists,
                                                y_hists,
                                                zs)
    self._update_weights_ops = self._update_weights(zs)
    self._update_history_ops = self._update_history(hist_idx,
                                                    hist_size,
                                                    zs,
                                                    Hzs,
                                                    s_hists,
                                                    y_hists)

  def _compute_step(self, hist_idx, hist_size, grads, s_hists, y_hists, zs):
    ops = []

    # Create tensors to compute dot products from the histories of s and y.
    sTy_hist = []
    yTy_hist = []
    # We construct the list so that sTy_history[-i] will compute the dot product
    # (s_{k-i}, y_{k-i}) etc.
    for i in reversed(range(1, self._conf['hist'] + 1)):
      sTy = tf.zeros([])
      yTy = tf.zeros([])
      idx = tf.mod(hist_idx - i, self._conf['hist'])
      for s_hist, y_hist in zip(s_hists, y_hists):
        sTy += tf.reduce_sum(s_hist[idx]*y_hist[idx])
        yTy += tf.reduce_sum(y_hist[idx]*y_hist[idx])
      sTy_hist.append(sTy)
      yTy_hist.append(yTy)

    # Start with the negative gradient.
    ps = []
    for g in grads:
      ps.append(-g)

    # First stage of the update (alg. 3, step 2 in the paper)
    alphas = []

    # Create a TensorFlow group that bundles all updates from an iteration
    # within the first stage.
    for i in range(1, self._conf['hist'] + 1):
      idx = tf.mod(hist_idx - i, self._conf['hist'])
      # Compute coefficient alpha (alg. 3, step 2a).
      sTp = tf.zeros(shape=[])
      for p, s_hist in zip(ps, s_hists):
        sTp += tf.reduce_sum(s_hist[idx]*p)
      alpha = tf.cond(i <= hist_size,
                      lambda: sTp/sTy_hist[-i],
                      lambda: tf.zeros(shape=[]))
      alphas.append(alpha)
      # Update direction (alg. 3, step 2b).
      for j, y_hist in enumerate(y_hists):
        ps[j] -= alpha*y_hist[idx]

    # Second stage of the update (alg. 3, step 3 and eq. 14)
    coeff = tf.zeros(shape=[])
    for i in range(1, self._conf['hist'] + 1):
      coeff = tf.cond(i <= hist_size,
                      lambda: coeff + sTy_hist[-i]/yTy_hist[-i],
                      lambda: coeff)
    coeff = tf.cond(tf.equal(hist_size, 0),
                    lambda: self._conf['eps'],
                    lambda: coeff/tf.cast(hist_size, dtype=coeff.dtype))
    for j in range(len(ps)):
      ps[j] *= coeff

    # Third stage of the update (alg. 3, step 4)
    for i in reversed(range(1, self._conf['hist'] + 1)):
      idx = tf.mod(hist_idx - i, self._conf['hist'])
      yTp = tf.zeros(shape=[])
      for p, y_hist in zip(ps, y_hists):
        yTp += tf.reduce_sum(y_hist[idx]*p)
      beta = yTp/sTy_hist[-i]
      alpha_minus_beta = tf.cond(i <= hist_size,
                                 lambda: alphas[i - 1] - beta,
                                 lambda: tf.zeros(shape=[]))
      for j, s_hist in enumerate(s_hists):
        ps[j] += alpha_minus_beta*s_hist[idx]

    # Save update.
    for p, z in zip(ps, zs):
      s = self._conf['lr']*p
      # We store the update in z, as we need to multiply it with the Hessian or
      # with the Gauss-Newton matrix in the next step.
      ops.append(tf.assign(z, s))

    return tf.group(ops)

  def _update_weights(self, zs):
    ops = []

    # Update weights.
    for w, z in zip(self._weights, zs):
      ops.append(tf.assign_add(w, z))

    return tf.group(ops)

  def _update_history(self, hist_idx, hist_size, zs, Hzs, s_hists, y_hists):
    ops = []

    for z, Hz, s_hist, y_hist in zip(zs, Hzs, s_hists, y_hists):
      # Use the Hessian or the Gauss-Newton approximation instead of a simple
      # difference of gradients.
      ops.append(tf.scatter_update(s_hist, hist_idx, z))
      ops.append(tf.scatter_update(y_hist, hist_idx, Hz))

    hist_idx_new = tf.mod(hist_idx + 1, self._conf['hist'])
    hist_size_new = tf.minimum(hist_size + 1, self._conf['hist'])

    with tf.control_dependencies(ops):
      ops.append(tf.assign(hist_idx, hist_idx_new))
      ops.append(tf.assign(hist_size, hist_size_new))

    return tf.group(ops)

  def get_gradient_aggregator(self):
    return self._grad_agg

  @staticmethod
  def get_default_conf():
    """
    Obtain the default optimizer configuration.
    The configuration is a Python dictionary with the following entries:

      'lr':
        Learning rate (default: 1.0).
      'eps':
        Small positive constant to ensure that the update in the first iteration
        is sufficiently small (default: 1.0e-4).
      'hist':
        Number of previous iterations to consider in the LBFGS update
        (default: 5).
      'reg':
        Non-negative regularization parameter (default: 0.0).
      'gauss_newton':
        Determines whether the Gauss-Newton approximation is used instead of the
        true Hessian.
        Note that without the Gauss-Newton approximation, this method is not
        likely to work for non-convex loss functions.
        However, the Gauss-Newton approximation only works if the loss function
        takes the form of the mean square error.
        (default: False)
      'num_samples':
        (Used if and only if gauss_newton is set to True.)
        The number of samples or batch size that will be used when evaluating
        the action of the Gauss-Newton matrix.
        (default: None)
      'pred':
        (Used if and only if gauss_newton is set to True.)
        TF tensor for the computation of the neural network output h(x, w) given
        data x and weights w.
        More specifically, pred[i] must contain the prediction h(x_i, w), i.e.,
        the prediction for the ith data sample.
        (default: None)
      'grad_agg':
        Name of the gradient aggregator to be used for variance reduction
        (default: 'raw_grad').
      'update_interval':
        If set to L > 1, the Hessian inverse approximation is updated once every
        L iterations (default: 1).
    """
    conf = {}
    conf['lr'] = 1.0
    conf['eps'] = 1.0e-4
    conf['hist'] = 5
    conf['reg'] = 0.0
    conf['gauss_newton'] = False
    conf['num_samples'] = None
    conf['pred'] = None
    conf['grad_agg'] = 'raw_grad'
    conf['update_interval'] = 1

    return conf

  def minimize(self, samples, samples_hessian=None):
    """See 'Optimizer' class."""
    if samples_hessian is None:
      samples_hessian = samples

    self._grad_agg.compute_grad(samples)
    self._sess.run(self._compute_step_ops)
    self._sess.run(self._update_weights_ops)
    if self._iter%self._conf['update_interval'] == 0:
      self._sess.run(self._update_history_ops, feed_dict=samples_hessian)

    self._iter += 1
