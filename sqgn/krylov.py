"""
krylov.py - Implementation of Krylov solvers.

Author: Christopher Thiele
"""

import math
import tensorflow as tf

class Solver:
  """Abstract base class for all linear solvers."""
  def __init__(self, Azs, zs, bs, xs, conf=None, sess=None):
    """
    Create a linear solver.

    Arguments:
      Azs:
        List of TF tensors that describe the action of an operator A.
      zs:
        List of TF variables that are the inputs for the evaluation of Azs.
      bs:
        Right-hand side of the linear system.
        Must be a list of tensors in the same format as xs.
      xs:
        List of TF variables to store the solution.
        Will be used as the initial guess for the solver unless specified
        otherwise in the solver configuration.
      conf:
        Linear solver configuration.
        If 'None', the default configuration is used.
      sess:
        TF session to be used by the solver.
        If 'None', the default session is used.

    Note that while the solution tensors and right-hand side are already passed
    when creating the solver, their values may be changed before or between
    calls to 'solve()'.
    """
    self._Azs = Azs
    self._zs = zs
    self._bs = bs
    self._xs = xs
    self._conf = conf

    if sess is None:
      self._sess = tf.get_default_session()
    else:
      self._sess = sess

  @property
  def conf(self):
    """Solver configuration."""
    return self._conf

  @property
  def sess(self):
    """TF session."""
    return self._sess

  def solve(self, feed_dict):
    """
    Solve a linear system.

    feed_dict:
      TF feed dictionary to be used when evaluating the operator Azs that was
      passed when the solver was created.
    """
    assert False, \
           "Cannot solve a linear system with the abstract solver base class!"
    
class CG(Solver):
  def __init__(self, Azs, zs, bs, xs, conf=None, sess=None):
    """See 'Solver' class."""
    if conf is None:
      conf = CG.get_default_conf()

    super().__init__(Azs, zs, bs, xs, conf, sess)

    # Create variables needed for the CG iteration.
    dtype = self._xs[0].dtype
    shapes = [x.shape.as_list() for x in self._xs]
    # Residual.
    rs = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]
    # Search direction.
    ps = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]
    # Squared residual norm(s).
    self._rTr = tf.Variable(tf.zeros(shape=[], dtype=dtype))
    self._rTr_prev = tf.Variable(tf.zeros(shape=[], dtype=dtype))

    self._indefinite = tf.Variable(False)

    self._set_initial_guess_ops = self._set_initial_guess()
    self._prepare_solve_ops = self._prepare_solve(rs, ps)
    self._compute_rTr_ops = self._compute_rTr(rs)
    self._set_operator_input_ops = self._set_operator_input(ps)
    self._update_ops = self._update(rs, ps)
    self._update_direction_ops = self._update_direction(rs, ps)

  def _set_initial_guess(self):
    ops = []

    if self._conf['zero_guess']:
      for x, z in zip(self._xs, self._zs):
        zero = tf.zeros(shape=x.shape, dtype=x.dtype)
        ops.append(tf.assign(x, zero))
        ops.append(tf.assign(z, zero))
    else:
      for x, z in zip(self._xs, self._zs):
        ops.append(tf.assign(z, x))

    return tf.group(ops)

  def _prepare_solve(self, rs, ps):
    ops = []

    if self._conf['zero_guess']:
      for r, b, p in zip(rs, self._bs, ps):
        # Set initial residual.
        ops.append(tf.assign(r, b))
        # Set initial search direction.
        ops.append(tf.assign(p, b))
    else:
      for r, b, p, Az in zip(rs, self._bs, ps, self._Azs):
        # Set initial residual.
        ops.append(tf.assign(r, b - Az))
        # Set initial search direction.
        ops.append(tf.assign(p, b - Az))

    ops.append(tf.assign(self._rTr, tf.zeros(shape=[], dtype=rs[0].dtype)))

    ops.append(tf.assign(self._indefinite, False))

    return tf.group(ops)

  def _compute_rTr(self, rs):
    ops = []

    save_rTr_prev = tf.assign(self._rTr_prev, self._rTr)
    ops.append(save_rTr_prev)

    with tf.control_dependencies([save_rTr_prev]):
      rTr = tf.zeros(shape=[], dtype=self._rTr.dtype)
      for r in rs:
        rTr += tf.reduce_sum(r*r)

      ops.append(tf.assign(self._rTr, rTr))

    return tf.group(ops)

  def _set_operator_input(self, ps):
    ops = []

    for z, p in zip(self._zs, ps):
      ops.append(tf.assign(z, p))

    return tf.group(ops)

  def _update(self, rs, ps):
    ops = []

    # Compute the coefficient alpha.
    pTAp = tf.zeros(shape=[], dtype=ps[0].dtype)
    for p, Az in zip(ps, self._Azs):
      # Recall that p has already been assigned to z, and hence Az = Ap.
      pTAp += tf.reduce_sum(p*Az)

    indefinite = pTAp <= 0.0
    ops.append(tf.assign(self._indefinite, indefinite))

    alpha = tf.cond(indefinite, lambda:0.0, lambda: self._rTr/pTAp)

    # Update the solution and residual.
    for x, r, p, Az in zip(self._xs, rs, ps, self._Azs):
      ops.append(tf.assign_add(x, alpha*p))
      ops.append(tf.assign_sub(r, alpha*Az))

    return tf.group(ops)

  def _update_direction(self, rs, ps):
    ops = []

    # Compute the coefficient beta.
    beta = self._rTr/self._rTr_prev

    # Update the search direction.
    for r, p in zip(rs, ps):
      ops.append(tf.assign(p, r + beta*p))

    return tf.group(ops)

  @staticmethod
  def get_default_conf():
    """
    Obtain the default solver configuration.
    The configuration is a Python dictionary with the following entries:

      'tol':
        Relative tolerance to be used as a stopping criterion (default: 1.0e-3).
      'maxiter':
        Maximum number of solver iterations (default: 1e3).
      'zero_guess':
        Whether or not to use zero as the initial guess (default: False).
    """
    conf = {}
    conf['tol'] = 1.0e-3
    conf['maxiter'] = 1e3
    conf['zero_guess'] = False

    return conf

  def solve(self, feed_dict):
    """See 'Solver' class."""
    self._sess.run(self._set_initial_guess_ops)
    self._sess.run(self._prepare_solve_ops, feed_dict=feed_dict)
    # Compute initial residual.
    # Run these operations twice to make sure the "previous" squared residual
    # norm is also initialized.
    self._sess.run(self._compute_rTr_ops)
    self._sess.run(self._compute_rTr_ops)
    r0norm = math.sqrt(self._rTr.eval(session=self._sess))

    k = 0
    while True:
      # Check residual.
      rnorm_rel = math.sqrt(self._rTr.eval(session=self._sess))/r0norm
      if rnorm_rel < self._conf['tol']:
        return

      if self._indefinite.eval(session=self._sess):
        return

      # Check number of iterations.
      if k >= self._conf['maxiter']:
        print('CG solver did not converge (||r||/||r0|| = %e)!' \
              % rnorm_rel)
        return

      self._sess.run(self._set_operator_input_ops)
      self._sess.run(self._update_ops, feed_dict=feed_dict)
      self._sess.run(self._compute_rTr_ops)
      self._sess.run(self._update_direction_ops)

      k += 1

class TRCG:
  """
  Trust region conjugate gradient solver.

  This method is a special version of CG that minimizes a local second-order
  approximation to the loss function within a trust region.
  """
  def __init__(self, hessians, zs, grads, dws, conf=None, sess=None):
    """
    Create a trust region CG solver.

    Arguments:
      hessians:
        List of TF tensors that describe the action of the Hessian of the loss
        function at the center of the trust region.
      zs:
        List of TF variables that are the inputs for the evaluation of the
        Hessian.
      grads:
        Gradients of the loss function at the center of the trust region.
        Must be a list of tensors in the same format as xs.
      dws:
        List of TF variables to store the solution.
      conf:
        Solver configuration.
        If 'None', the default configuration is used.
      sess:
        TF session to be used by the solver.
        If 'None', the default session is used.

    Note that this solver will always start with an initial guess of zero,
    regardless of the iniial value of dws.
    """
    if conf is None:
      conf = TRCG.get_default_conf()

    if sess is None:
      sess = tf.get_default_session()

    self._hessians = hessians
    self._zs = zs
    self._grads = grads
    self._dws = dws
    self._conf = conf
    self._sess = sess

    # Create variables needed for the CG iteration.
    dtype = self._dws[0].dtype
    shapes = [dw.shape.as_list() for dw in self._dws]
    # Residual.
    rs = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]
    # Search direction.
    ps = [tf.Variable(tf.zeros(shape=s, dtype=dtype)) for s in shapes]
    # Squared residual norm(s).
    self._rTr = tf.Variable(tf.zeros(shape=[], dtype=dtype))
    self._rTr_prev = tf.Variable(tf.zeros(shape=[], dtype=dtype))

    # Create a tensor to compute the norm of the current iterate.
    self._norm_dw = tf.zeros(shape=[], dtype=dtype)
    for dw, p in zip(self._dws, ps):
      self._norm_dw += tf.reduce_sum(dw*dw)
    self._norm_dw = tf.sqrt(self._norm_dw)

    # Create a placeholder for the trust region radius, which is unknown at this
    # time.
    self._radius_placeh = tf.placeholder(shape=[], dtype=dtype)

    self._zero_dw_ops = self._zero_dws()
    self._prepare_solve_ops = self._prepare_solve(rs, ps)
    self._compute_rTr_ops = self._compute_rTr(rs)
    self._set_operator_input_ops = self._set_operator_input(ps)
    self._update_ops = self._update(rs, ps)
    self._update_direction_ops = self._update_direction(rs, ps)
    self._follow_p_ops = self._follow_p(ps)

  def _zero_dws(self):
    ops = []

    for dw in self._dws:
      ops.append(tf.assign(dw, tf.zeros(shape=dw.shape, dtype=dw.dtype)))

    return tf.group(ops)

  def _prepare_solve(self, rs, ps):
    ops = []

    for g, r, p in zip(self._grads, rs, ps):
      # Set initial residual.
      ops.append(tf.assign(r, -g))
      # Set initial search direction.
      ops.append(tf.assign(p, -g))

    ops.append(tf.assign(self._rTr, tf.zeros(shape=[], dtype=rs[0].dtype)))

    return tf.group(ops)

  def _compute_rTr(self, rs):
    ops = []

    save_rTr_prev = tf.assign(self._rTr_prev, self._rTr)
    ops.append(save_rTr_prev)

    with tf.control_dependencies([save_rTr_prev]):
      rTr = tf.zeros(shape=[], dtype=self._rTr.dtype)
      for r in rs:
        rTr += tf.reduce_sum(r*r)

      ops.append(tf.assign(self._rTr, rTr))

    return tf.group(ops)

  def _set_operator_input(self, ps):
    ops = []

    for z, p in zip(self._zs, ps):
      ops.append(tf.assign(z, p))

    return tf.group(ops)

  def _update(self, rs, ps):
    ops = []

    # Compute the coefficient alpha.
    pTHp = tf.zeros(shape=[], dtype=ps[0].dtype)
    for p, Hz in zip(ps, self._hessians):
      # Recall that p has already been assigned to z, and hence Hz = Hp.
      pTHp += tf.reduce_sum(p*Hz)
      
    # Compute the coefficient for the update.
    alpha = self._rTr/pTHp

    # Create a tensor that computes the norm of the iterate after the update
    # without actually modifying it.
    norm_dw_new = tf.zeros(shape=[], dtype=self._norm_dw.dtype)
    for dw, p in zip(self._dws, ps):
      dw_new = dw + alpha*p
      norm_dw_new += tf.reduce_sum(dw_new*dw_new)
    norm_dw_new = tf.sqrt(norm_dw_new)

    # Determine if we should follow the direction p until it intersects with the
    # boundary of the trust region.
    # This is the case if either p is a direction of indefiniteness or if dw + p
    # would be outside the trust region.
    follow_to_boundary = tf.logical_or(pTHp <= 0.0,
                                       norm_dw_new > self._radius_placeh)
    self._follow_to_boundary = tf.Variable(False)
    ops.append(tf.assign(self._follow_to_boundary, follow_to_boundary))

    # If we follow p up to the boundary, we do not update dw here.
    # Instead, we determine the final update dw in the 'solve' method.
    alpha_or_zero = tf.cond(follow_to_boundary,
                            lambda: 0.0,
                            lambda: alpha)

    # Update the solution and residual.
    for dw, r, p, Hz in zip(self._dws, rs, ps, self._hessians):
      ops.append(tf.assign_add(dw, alpha_or_zero*p))
      ops.append(tf.assign_sub(r, alpha_or_zero*Hz))

    return tf.group(ops)

  def _update_direction(self, rs, ps):
    ops = []

    # Compute the coefficient beta.
    beta = self._rTr/self._rTr_prev

    # Update the search direction.
    for r, p in zip(rs, ps):
      ops.append(tf.assign(p, r + beta*p))

    return tf.group(ops)

  def _follow_p(self, ps):
    ops = []

    dtype = self._dws[0].dtype
    pTdw = tf.zeros(shape=[], dtype=dtype)
    pTp = tf.zeros(shape=[], dtype=dtype)
    dwTdw = tf.zeros(shape=[], dtype=dtype)
    for p, dw in zip(ps, self._dws):
      pTdw += tf.reduce_sum(p*dw)
      pTp += tf.reduce_sum(p*p)
      dwTdw += tf.reduce_sum(dw*dw)

    tau = (-pTdw + tf.sqrt(pTdw**2 + pTp*(self._radius_placeh**2 - dwTdw)))/pTp

    for p, dw in zip(ps, self._dws):
      dw_new = dw + tau*p
      ops.append(tf.assign(dw, dw_new))

    return tf.group(ops)

  @staticmethod
  def get_default_conf():
    """
    Obtain the default solver configuration.
    The configuration is a Python dictionary with the following entries:

      'tol':
        Relative tolerance to be used as a stopping criterion (default: 1.0e-3).
      'maxiter':
        Maximum number of solver iterations (default: 1e3).
    """
    conf = {}
    conf['tol'] = 1.0e-3
    conf['maxiter'] = 1e3

    return conf

  def solve(self, radius, samples, samples_hessian=None):
    """
    Solve the trust region subproblem.

    Arguments:
      radius:
        Radius of the trust region.
      samples:
        TF feed dictionary to be used when evaluating gradients.
      samples_hessian:
        TF feed dictionary to be used when evaluating Hessians.
        If 'None', the same dictionary is used that is also used for the
        gradients.

    Returns:
      Norm of the computed update.
    """
    self._sess.run(self._zero_dw_ops)
    self._sess.run(self._prepare_solve_ops, feed_dict=samples)
    # Compute initial residual.
    # Run these operations twice to make sure the "previous" squared residual
    # norm is also initialized.
    self._sess.run(self._compute_rTr_ops)
    self._sess.run(self._compute_rTr_ops)
    r0norm = math.sqrt(self._rTr.eval(session=self._sess))

    # Prepare feed dictionary for the update step.
    update_dict = samples_hessian
    update_dict[self._radius_placeh] = radius

    k = 0
    while True:
      # Check residual.
      rnorm_rel = math.sqrt(self._rTr.eval(session=self._sess))/r0norm
      if rnorm_rel < self._conf['tol']:
        break

      # Check number of iterations.
      if k >= self._conf['maxiter']:
        print('TRCG solver did not converge (||r||/||r0|| = %e)!' \
              % rnorm_rel)
        break

      self._sess.run(self._set_operator_input_ops)
      self._sess.run(self._update_ops, feed_dict=update_dict)

      if self._follow_to_boundary.eval(session=self._sess):
        # Follow the direction p up to the boundary of the trust region and then
        # stop.
        self._sess.run(self._follow_p_ops,
                       feed_dict={self._radius_placeh: radius})
        break

      self._sess.run(self._compute_rTr_ops)
      self._sess.run(self._update_direction_ops)

      k += 1
