"""
newton.py - Newton-type methods.

Author: Christopher Thiele
"""

import sqgn
from sqgn.optimizer import Optimizer

import tensorflow as tf
import numpy as np

class Newton(Optimizer):
  """Abstract base class for all Newton-type optimizers."""
  def __init__(self, loss, weights, conf=None, sess=None):
    """See 'Optimizer' class."""
    super().__init__(loss, weights, conf, sess)

    # Compute gradients.
    self._grads = tf.gradients(self._loss, self._weights)
    
    dtype = self._loss.dtype
    shapes = [w.shape.as_list() for w in self._weights]

    # Now we need to create tensors that describe the action of the Hessian.
    # We exploit that grad((grad(f), z)) = Hz, where H is the Hessian of f and z
    # is a vector that does not depend on the variables w.r.t. which the
    # gradient is taken.
    # Since the inner product (grad(f), z) is a scalar quantity, this way of
    # computing Hp is much more efficient than computing the full Hessian H
    # first and then multiplying it with z.
    self._zs = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]
    grad_dot_z = tf.zeros(shape=[], dtype=dtype)
    for g, z in zip(self._grads, self._zs):
      grad_dot_z += tf.reduce_sum(g*z)
    Hzs = tf.gradients(grad_dot_z, self._weights)
    self._hessians = []
    for z, Hz in zip(self._zs, Hzs):
      self._hessians.append(Hz + self._conf['reg']*z)

  def minimize(self, samples, samples_hessian=None):
    """See 'Optimizer' class."""
    assert False, 'Cannot minimize with abstract Newton method base class!'

class NewtonCG(Newton):
  """Inexact Newton method that uses CG to approximate the inverse Hessian."""
  def __init__(self, loss, weights, conf=None, sess=None):
    """See 'Optimizer' class."""
    if conf is None:
      conf = NewtonCG.get_default_conf()

    super().__init__(loss, weights, conf, sess)

    dtype = self._loss.dtype
    shapes = [w.shape.as_list() for w in self._weights]

    # Create variables for weight updates.
    self._dws = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]

    # Create variables for the right-hand side of the linear system.
    # While the right-hand side is simply the gradient of the loss function, 
    # we cannot pass the gradient to the solver directly, because the solver
    # uses a different feed dictionary (Hessian subsampling).
    # Hence, we must evaluate the gradient and store a copy before the solve.
    self._rhss = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]

    # Create the linear solver.
    self._solver = sqgn.CG(self._hessians,
                           self._zs,
                           self._rhss,
                           self._dws,
                           conf=self._conf['solver_conf'],
                           sess=self._sess)

    # Set up the line search method.
    line_search_conf = self._conf['line_search_conf']
    self._line_search = sqgn.create_line_search(self._conf['line_search'],
                                                self._loss,
                                                self._weights,
                                                self._dws,
                                                self._rhss,
                                                conf=line_search_conf,
                                                sess=self._sess)

    self._prepare_solve_ops = self._prepare_solve()

  def _prepare_solve(self):
    ops = []

    for dw, rhs, g in zip(self._dws, self._rhss, self._grads):
      ops.append(tf.assign(dw, -g))
      ops.append(tf.assign(rhs, -g))

    return tf.group(ops)

  @staticmethod
  def get_default_conf():
    """
    Obtain the default optimizer configuration.
    The configuration is a Python dictionary with the following entries:

      'lr':
        Learning rate (default: 1.0).
      'reg':
        Regularization parameter (default: 0.0).
        When solving the linear system in each step, this coefficient times the
        identity is added to the operator to make it more definite.
      'solver_conf':
        CG solver configuration (default: default CG solver configuration with
        a nonzero initial guess).
      'line_search':
        Name of the line search method to be used (default: 'full_step').
      'line_search_conf':
        Line search configuration (default: None).
    """
    conf = {}
    conf['lr'] = 1.0
    conf['reg'] = 0.0
    solver_conf = sqgn.krylov.CG.get_default_conf()
    solver_conf['zero_guess'] = False
    conf['solver_conf'] = solver_conf
    conf['line_search'] = 'full_step'
    conf['line_search_conf'] = None

    return conf

  def minimize(self, samples, samples_hessian=None):
    """See 'Optimizer' class."""
    if samples_hessian is None:
      samples_hessian = samples

    self._sess.run(self._prepare_solve_ops, feed_dict=samples)
    self._solver.solve(feed_dict=samples_hessian)
    self._line_search.minimize(samples)

class GaussNewtonCG(Newton):
  """Gauss-Newton method that uses CG to solve for the Newton update."""
  def __init__(self, loss, weights, conf=None, sess=None):
    """See 'Optimizer' class."""
    if conf is None:
      assert False, 'Cannot use Gauss-Newton method with default configuration!'
    if conf['num_samples'] is None:
      assert False, 'User must provide number of samples for Gauss-Newton!'
    if conf['pred'] is None:
      assert False, 'User must provide a residual function for Gauss-Newton!'

    super().__init__(loss, weights, conf, sess)

    dtype = self._loss.dtype
    shapes = [w.shape.as_list() for w in self._weights]

    # Create variables for weight updates.
    self._dws = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]

    # Create variables for the right-hand side of the linear system.
    # While the right-hand side is simply the gradient of the loss function, 
    # we cannot pass the gradient to the solver directly, because the solver
    # uses a different feed dictionary (Hessian subsampling).
    # Hence, we must evaluate the gradient and store a copy before the solve.
    self._rhss = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]

    # Create tensors that describe the action of the Gauss-Newton matrix.
    JTJzs = sqgn.gauss_newton.JTHJv(self._conf['pred'],
                                    self._loss,
                                    self._conf['num_samples'],
                                    self._weights,
                                    self._zs)

    # Add regularization.
    i = 0
    for JTJz, z in zip(JTJzs, self._zs):
      JTJzs[i] = JTJz + self._conf['reg']*z
      i += 1

    # Create the linear solver.
    self._solver = sqgn.CG(JTJzs,
                           self._zs,
                           self._rhss,
                           self._dws,
                           conf=self._conf['solver_conf'],
                           sess=self._sess)

    self._prepare_solve_ops = self._prepare_solve()
    self._update_weights_ops = self._update_weights()

  def _prepare_solve(self):
    ops = []

    for dw, rhs, g in zip(self._dws, self._rhss, self._grads):
      ops.append(tf.assign(dw, -g))
      ops.append(tf.assign(rhs, -g))

    return tf.group(ops)

  def _update_weights(self):
    ops = []

    for w, dw in zip(self._weights, self._dws):
      ops.append(tf.assign_add(w, self._conf['lr']*dw))

    return tf.group(ops)

  @staticmethod
  def get_default_conf():
    """
    Obtain the default optimizer configuration.
    The configuration is a Python dictionary with the following entries:

      'lr':
        Learning rate (default: 1.0).
      'reg':
        Regularization parameter (default: 0.0).
        When solving the linear system in each step, this coefficient times the
        identity is added to the operator to make it more definite.
      'solver_conf':
        CG solver configuration (default: default CG solver configuration with
        an initial guess of zero).
      'line_search':
        Name of the line search method to be used (default: 'full_step').
      'num_samples':
        The number of samples or batch size that will be used when evaluating
        the action of the Gauss-Newton matrix.
        This must be provided by the user.
        (default: None)
      'pred':
        (Used if and only if gauss_newton is set to True.)
        TF tensor for the computation of the neural network output h(x, w) given
        data x and weights w.
        More specifically, pred[i] must contain the prediction h(x_i, w), i.e.,
        the prediction for the ith data sample.
        (default: None)
    """
    conf = {}
    conf['lr'] = 1.0
    conf['reg'] = 0.0
    solver_conf = sqgn.krylov.CG.get_default_conf()
    solver_conf['zero_guess'] = True
    conf['solver_conf'] = solver_conf
    conf['num_samples'] = None
    conf['pred'] = None

    return conf

  def minimize(self, samples, samples_hessian=None):
    """See 'Optimizer' class."""
    if samples_hessian is None:
      samples_hessian = samples

    self._sess.run(self._prepare_solve_ops, feed_dict=samples)
    self._solver.solve(feed_dict=samples_hessian)
    self._sess.run(self._update_weights_ops)

class NewtonTR(Newton):
  """Inexact Newton method with trust regions."""
  def __init__(self, loss, weights, conf=None, sess=None):
    """See 'Optimizer' class."""
    if conf is None:
      conf = NewtonTR.get_default_conf()

    super().__init__(loss, weights, conf, sess)

    # Trust region radius cannot be determined at this time.
    self._radius = None

    dtype = self._loss.dtype
    shapes = [w.shape.as_list() for w in self._weights]

    # Create variables for weight updates.
    self._dws = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]

    # Create variables for the momentum.
    self._vs = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]
    self._vs_scaled = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) \
                       for s in shapes]

    # Create variables for the old weights.
    self._weights_prev = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) \
                          for s in shapes]

    # Create variables for another copy of the weights.
    # Note that _weights_prev will contain a copy of the weights before the call
    # to minimize(), while _weights_prev_radius will contain a copy of the
    # weights before the latest increase of the trust region radius.
    self._weights_prev_radius = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) \
                                 for s in shapes]

    # Create a tensor for the norms of the gradient and the updates.
    self._grad_norm = tf.zeros(shape=[], dtype=dtype)
    self._norm_dw = tf.zeros(shape=[], dtype=dtype)
    for g, dw in zip(self._grads, self._dws):
      self._grad_norm += tf.reduce_sum(g*g)
      self._norm_dw += tf.reduce_sum(dw*dw)
    self._grad_norm = tf.sqrt(self._grad_norm)
    self._norm_dw = tf.sqrt(self._norm_dw)

    # Create a TF placeholder for the trust region radius.
    self._radius_placeh = tf.placeholder(shape=[], dtype=dtype)

    # Create the tensors needed for the computation of the predicted loss.
    # Note that we have to create two separate tensors, because the gradient
    # and the Hessian use different samples.
    self._grad_dot_dw = tf.zeros(shape=[], dtype=dtype)
    self._dw_dot_Hz = tf.zeros(shape=[], dtype=dtype)
    for g, dw, Hz in zip(self._grads, self._dws, self._hessians):
      self._grad_dot_dw += tf.reduce_sum(g*dw)
      self._dw_dot_Hz += tf.reduce_sum(dw*Hz)

    # Create trust region CG solver.
    self._solver = sqgn.TRCG(self._hessians,
                             self._zs,
                             self._grads,
                             self._dws,
                             conf=self._conf['solver_conf'],
                             sess=self._sess)

    self._copy_old_weights_ops = self._copy_old_weights()
    self._update_weights_ops = self._update_weights()
    self._scale_momentum_ops = self._scale_momentum()
    self._apply_momentum_ops = self._apply_momentum()
    self._restrict_update_ops = self._restrict_update()
    self._undo_update_ops = self._undo_update()
    self._assign_dw_to_z_ops = self._assign_dw_to_z()
    self._save_weights_ops = self._save_weights()
    self._restore_weights_ops = self._restore_weights()
    self._update_momentum_ops = self._update_momentum()

  def _update_weights(self):
    ops = []

    for w, dw in zip(self._weights, self._dws):
      ops.append(tf.assign_add(w, dw))

    return tf.group(ops)

  def _undo_update(self):
    ops = []

    for w, dw in zip(self._weights, self._dws):
      ops.append(tf.assign_sub(w, dw))

    return tf.group(ops)

  def _assign_dw_to_z(self):
    ops = []

    for z, dw in zip(self._zs, self._dws):
      ops.append(tf.assign(z, dw))

    return tf.group(ops)

  def _copy_old_weights(self):
    ops = []

    for w, w_prev in zip(self._weights, self._weights_prev):
      ops.append(tf.assign(w_prev, w))

    return tf.group(ops)

  def _scale_momentum(self):
    ops = []

    norm_v = tf.zeros(shape=[], dtype=self._loss.dtype)
    for v in self._vs:
      norm_v += tf.reduce_sum(v*v)
    norm_v = tf.sqrt(norm_v)

    coeff = self._conf['beta']*tf.minimum(1.0, self._radius_placeh/norm_v)

    for v, v_scaled in zip(self._vs, self._vs_scaled):
      ops.append(tf.assign(v_scaled, coeff*v))

    return tf.group(ops)

  def _apply_momentum(self):
    ops = []

    # Apply the scaled momentum to the update.
    for dw, v_scaled in zip(self._dws, self._vs_scaled):
      ops.append(tf.assign_add(dw, v_scaled))

    return tf.group(ops)

  def _restrict_update(self):
    ops = []

    coeff = tf.minimum(1.0, self._radius_placeh/self._norm_dw)

    for dw in self._dws:
      ops.append(tf.assign(dw, coeff*dw))

    return tf.group(ops)

  def _save_weights(self):
    ops = []

    for w, w_prev_radius in zip(self._weights, self._weights_prev_radius):
      ops.append(tf.assign(w_prev_radius, w))

    return tf.group(ops)

  def _restore_weights(self):
    ops = []

    for w, w_prev_radius in zip(self._weights, self._weights_prev_radius):
      ops.append(tf.assign(w, w_prev_radius))

    return tf.group(ops)

  def _update_momentum(self):
    ops = []

    for v, v_scaled, w, w_prev \
    in zip(self._vs, self._vs_scaled, self._weights, self._weights_prev):
      ops.append(tf.assign(v, self._conf['beta']*v_scaled + w - w_prev))

    return tf.group(ops)

  @staticmethod
  def get_default_conf():
    """
    Obtain the default optimizer configuration.
    The configuration is a Python dictionary with the following entries:

      'reg':
        Regularization parameter (default: 0.0).
        When solving the linear system in each step, this coefficient times the
        identity is added to the operator to make it more definite.
      'beta':
        Coefficient for the momentum (default: 0.0).
        A value of 0.0 disables momentum, while positive values will add
        momentum to the weight updates.
      'increase':
        Factor by which the radius of the trust region increases
        (see below, default: 2.0).
      'decrease':
        Factor by which the radius of the trust region decreases
        (see below, default: 0.5).
      'thresholds':
        List of three numbers between zero and one that govern the behavior of
        the trust region method:
          - If the quotient of the actual and predicted reduction in loss is
            less that thresholds[0], the Newton step is rejected, and the radius
            of the trust region is reduced.
          - If the quotient is between thresholds[0] and thresholds[1], then the
            Newton step is accepted, but the radius is still reduced.
          - If the quotient is between thresholds[1] and thresholds[2], the
            Newton step is accepted, and the radius of the trust region remains
            unchanged.
          - If the quotient is greater than thresholds[2] and the norm of the
            update is equal to the radius of the trust region, the radius is
            increased, and a new Newton step is computed.
        (default: [1.0e-4, 0.25, 0.75]).
      'reset':
        If 'True', the radius of the trust region is reset at the beginning of
        each Newton iteration, i.e., at the beginning of each mini-batch
        (default: False).
      'solver_conf':
        Configuration for the trust region subproblem solver
        (default: default TRCG configuration).
    """
    conf = {}
    conf['reg'] = 0.0
    conf['beta'] = 0.0
    conf['increase'] = 2.0
    conf['decrease'] = 0.5
    conf['thresholds'] = [1.0e-4, 0.25, 0.75]
    conf['reset'] = False
    conf['solver_conf'] = sqgn.krylov.TRCG.get_default_conf()

    return conf

  def minimize(self, samples, samples_hessian=None):
    """See 'Optimizer' class."""
    if samples_hessian is None:
      samples_hessian = samples

    if self._radius is None:
      # Initialize trust region radius based on the norm of the gradient.
      self._radius = self._grad_norm.eval(session=self._sess, feed_dict=samples)

    radius_increased = False

    # Save the current weights before any updates.
    self._sess.run(self._copy_old_weights_ops)

    while True:
      # Solve trust region problem.
      self._solver.solve(self._radius,
                         samples,
                         samples_hessian=samples_hessian)

      # Scale the momentum to the current size of the trust region and apply it
      # to the update.
      self._sess.run(self._scale_momentum_ops,
                     feed_dict={self._radius_placeh: self._radius})
      self._sess.run(self._apply_momentum_ops)
      self._sess.run(self._restrict_update_ops,
                     feed_dict={self._radius_placeh: self._radius})

      # Check if the update is in the interior of the trust region or on the
      # boundary.
      norm_dw = self._norm_dw.eval(session=self._sess)
      update_on_boundary = norm_dw >= self._radius - 1.0e-6

      # Keep the current loss for later.
      actual_reduction = self._loss.eval(session=self._sess, feed_dict=samples)

      # Compute predicted reduction in loss.
      predicted_reduction = -self._grad_dot_dw.eval(session=self._sess,
                                                    feed_dict=samples)
      self._sess.run(self._assign_dw_to_z_ops)
      predicted_reduction -= 0.5*self._dw_dot_Hz.eval(session=self._sess,
                                                      feed_dict=samples_hessian)

      # Update weights and compute actual reduction in loss.
      self._sess.run(self._update_weights_ops)
      actual_reduction -= self._loss.eval(session=self._sess, feed_dict=samples)

      # Decide how to proceed based on the ratio of actual and predicted
      # reduction in the loss.
      ratio = actual_reduction/predicted_reduction
      if ratio < self._conf['thresholds'][0]:
        # Reject update and decrease radius.
        self._radius *= self._conf['decrease']
        self._sess.run(self._undo_update_ops)
        if radius_increased:
          # The radius has just been increased, but now it is too large.
          # Hence, we accept the old weights and stop.
          self._sess.run(self._restore_weights_ops)
          radius_increased = False
          break
        radius_increased = False
      elif ratio < self._conf['thresholds'][1]:
        # Accept weights, but reduce radius.
        self._radius *= self._conf['decrease']
        break
      elif ratio < self._conf['thresholds'][2]:
        # Leave radius unchanged and accept weights.
        break
      elif ratio >= self._conf['thresholds'][2] and update_on_boundary:
        # Reject update and increase trust region.
        self._radius *= self._conf['increase']
        # Keep a copy of the current weights in case the iteration with the
        # larger trust region fails.
        self._sess.run(self._save_weights_ops)
        # Revert the weights *after* saving them in order to solve the trust
        # region problem again, starting at the same point, but with increased
        # radius.
        self._sess.run(self._undo_update_ops)
        radius_increased = True
      else:
        # The only remaining case is the situation where the update is inside
        # the trust region and the actual reduction is higher than predicted.
        # In this case we accept the update and do not increase the trust
        # region.
        break
        
    # Update the momentum.
    self._sess.run(self._update_momentum_ops)

    if self._conf['reset']:
      # Make sure that a new trust region radius is determined at the beginning
      # of the next iteration.
      self._radius = None

class CurveBall(Newton):
  """CurveBall method."""
  def __init__(self, loss, weights, conf=None, sess=None):
    """See 'Optimizer' class."""
    if conf is None:
      conf = CurveBall.get_default_conf()

    super().__init__(loss, weights, conf, sess)

    dtype = self._loss.dtype
    shapes = [w.shape.as_list() for w in self._weights]

    # Create the tensors needed for the CurveBall algorithm.
    self._ys = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]
    self._dys = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]
    self._Hys = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]

    # Create placeholders for the coefficients rho and beta.
    self._rho = tf.placeholder(shape=[], dtype=dtype)
    self._beta = tf.placeholder(shape=[], dtype=dtype)

    # Create tensors required for the automatic computation of rho and beta.
    self._grad_dot_y = tf.zeros(shape=[], dtype=dtype)
    self._grad_dot_dy = tf.zeros(shape=[], dtype=dtype)
    self._y_dot_Hy = tf.zeros(shape=[], dtype=dtype)
    self._dy_dot_Hy = tf.zeros(shape=[], dtype=dtype)
    for g, y, dy, Hy in zip(self._grads, self._ys, self._dys, self._Hys):
      self._grad_dot_y += tf.reduce_sum(g*y)
      self._grad_dot_dy += tf.reduce_sum(g*dy)
      self._y_dot_Hy += tf.reduce_sum(y*Hy)
      self._dy_dot_Hy += tf.reduce_sum(dy*Hy)
    self._dy_dot_Hdy = tf.zeros(shape=[], dtype=dtype)
    for dy, Hz in zip(self._dys, self._hessians):
      self._dy_dot_Hdy += tf.reduce_sum(dy*Hz)

    self._assign_y_to_z_ops = self._assign_y_to_z()
    self._eval_Hy_ops = self._eval_Hy()
    self._update_dy_ops = self._update_dy()
    self._assign_dy_to_z_ops = self._assign_dy_to_z()
    self._update_y_ops = self._update_y()
    self._update_weights_ops = self._update_weights()

    # Iteration counter
    self._k = 0

  def _assign_y_to_z(self):
    ops = []

    for z, y in zip(self._zs, self._ys):
      ops.append(tf.assign(z, y))

    return tf.group(ops)

  def _eval_Hy(self):
    ops = []

    for Hz, Hy, dy in zip(self._hessians, self._Hys, self._dys):
      ops.append(tf.assign(Hy, Hz))
      ops.append(tf.assign(dy, Hz))

    return tf.group(ops)

  def _update_dy(self):
    ops = []

    for dy, g in zip(self._dys, self._grads):
      ops.append(tf.assign_add(dy, g))

    return tf.group(ops)

  def _assign_dy_to_z(self):
    ops = []

    for z, dy in zip(self._zs, self._dys):
      ops.append(tf.assign(z, dy))

    return tf.group(ops)

  def _update_y(self):
    ops = []

    for y, dy in zip(self._ys, self._dys):
      ops.append(tf.assign(y, self._rho*y - self._beta*dy))

    return tf.group(ops)

  def _update_weights(self):
    ops = []

    for w, y in zip(self._weights, self._ys):
      ops.append(tf.assign_add(w, self._conf['lr']*y))

    return tf.group(ops)

  @staticmethod
  def get_default_conf():
    """
    Obtain the default optimizer configuration.
    The configuration is a Python dictionary with the following entries:

      'lr':
        Learning rate (default: 1.0).
      'reg':
        Regularization parameter (default: 0.0).
        When solving the linear system in each step, this coefficient times the
        identity is added to the operator to make it more definite.
    """
    conf = {}
    conf['lr'] = 1.0
    conf['reg'] = 0.0

    return conf

  def minimize(self, samples, samples_hessian=None):
    """See 'Optimizer' class."""
    if samples_hessian is None:
      samples_hessian = samples

    self._k = self._k + 1

    self._sess.run(self._assign_y_to_z_ops)
    self._sess.run(self._eval_Hy_ops, feed_dict=samples_hessian)
    self._sess.run(self._update_dy_ops, feed_dict=samples)
    self._sess.run(self._assign_dy_to_z_ops)

    # Solve a 2x2-system for the optimal values of rho and beta.
    gTy, gTdy = self._sess.run([self._grad_dot_y, self._grad_dot_dy],
                               feed_dict=samples)
    yTHy, yTHdy, dyTHdy = self._sess.run(
                            [self._y_dot_Hy, self._dy_dot_Hy, self._dy_dot_Hdy],
                            feed_dict=samples_hessian
                          )
    if self._k == 1:
      # The linear system below is singular during the first iteration.
      # Instead, we solve for beta directly, since the coefficient rho does not
      # matter in the first iteration, because y = 0.
      rho = 0.0
      beta = gTdy/dyTHdy
    else:
      A = np.array([[dyTHdy, yTHdy], [yTHdy, yTHy]])
      b = np.array([-gTdy, -gTy])
      coeffs = np.linalg.solve(A, b)
      rho = coeffs[1]
      beta = -coeffs[0]

    rho_and_beta = {self._rho: rho, self._beta: beta}
    self._sess.run(self._update_y_ops, feed_dict=rho_and_beta)
    self._sess.run(self._update_weights_ops)
