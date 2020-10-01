"""
gauss_newton.py - Shared functions for the implementation of Gauss-Newton
                  methods in TensorFlow.

Author: Christopher Thiele
"""

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def JTJv(residual, n, xs, vs):
  """
  Returns a tensor that represents the product of J^TJ with v, where J is the
  Jacobian of the mean square of the residual w.r.t. the given variables.
  More specifically, J is the Jacobian of reduce_sum(residual**2)/n.

  Arguments:
    residual:
      The residual tensor.
    n:
      Extent of the residual tensor in the 0th dimension at the time when JTJv
      is evaluated.
      The shape of the residual tensor might be undetermined, e.g., if the
      residual is derived from a neural network with a variable batch size.
      In this case, a value for the batch size n must still be provided here for
      the purpose of memory allocation.
      By doing so, the user guarantees that the batch size will be equal to n
      whenever the tensor returned by this function is evaluated.
    xs:
      List of variables w.r.t. which the Jacobian is formed.
    vs:
      List of tensors with which J^TJ is multiplied.
      Each tensor in the list must have the same shape as the corresponding
      variable in xs.
  """

  # First, we reshape the residual to a vector so we can formulate the algorithm
  # in the familiar setting of vectors and matrices.
  r = tf.reshape(residual, [-1])

  # Now we create a variable u for the definition of J^Tu, i.e., in order to
  # define the action of the transpose of the Jacobian.
  # Note that the value of u is irrelevant at all times, but unfortunately it
  # was not possible to implement the functionality of the code below using a
  # placeholder as of 07/10/2019.
  shape = residual.shape.as_list()
  shape[0] = n
  length = 1
  for s in shape:
    length *= s
  u = tf.Variable(tf.zeros(shape=[length], dtype=residual.dtype))

  # Next, we define the action of the transpose of the Jacobian on the vector u.
  # Here, we use the trick that the gradient of the inner product (a, b) is
  # equal to J_a^Tb + J_b^Ta, where J_a and J_b are the Jacobians of a and b
  # respectively.
  # Hence, if b is a constant, we simply obtain J_a^Tb.
  JTus = tf.gradients(tf.reduce_sum(r*u), xs)

  # Now we can define the action of the Jacobian itself.
  # For reasons of efficiency, we must again define it as the gradient of a
  # scalar function using the same trick as before.
  # This time however, we take the gradient of (J^Tu, v) w.r.t. u, and we obtain
  # (J^T)^Tv = Jv as desired.
  # For more detail on this procedure, please refer to
  # https://j-towns.github.io/2017/06/12/A-new-trick.html.
  # Note that the value of u becomes irrelevant at this point.
  JTus_dot_vs = [tf.reduce_sum(JTu*v) for JTu, v in zip(JTus, vs)]
  Jv = tf.gradients(JTus_dot_vs, u)[0]

  # At this point we can define the action of J^TJ on the vector v.
  # The definition is identical to that of JTu above, but this time we use Jv
  # instead of u.
  # Note that we have to treat Jv as a constant here, as it still depends on x.
  JTJvs = tf.gradients(tf.reduce_sum(r*Jv), xs, stop_gradients=Jv)

  # So far we have computed the action of the Gauss-Newton matrix of
  # reduce_sum(residual**2), and hence we still need to scale the result
  # correctly.
  for i, JTJv in enumerate(JTJvs):
    JTJvs[i] = JTJv/(n**2)

  return JTJvs

def JTHJv(y, l, n, xs, vs):
  """
  Returns a tensor that represents the action of the Gauss-Newton operator
  J^THJ on v.
  Here, J is the Jacobian of y w.r.t. x, and H is the Hessian of l w.r.t. y.
  Hence, J^THJ is the positive semidefinite Gauss-Newton approximation to the
  Hessian of l(y(x)) w.r.t. x

  Arguments:
    y:
      Inner function, typically the output of a neural network.
    l:
      Outer function, typically the loss as a function of y.
    n:
      Extent of the residual tensor in the 0th dimension at the time when JTHJv
      is evaluated.
      The shape of the residual tensor might be undetermined, e.g., if the
      residual is derived from a neural network with a variable batch size.
      In this case, a value for the batch size n must still be provided here for
      the purpose of memory allocation.
      By doing so, the user guarantees that the batch size will be equal to n
      whenever the tensor returned by this function is evaluated.
    xs:
      List of variables w.r.t. which the Jacobian is formed.
    vs:
      List of tensors with which J^THJ is multiplied.
      Each tensor in the list must have the same shape as the corresponding
      variable in xs.
  """

  # Please see JTJv for a description of the procedure.

  # Create auxiliary variable.
  shape = y.shape.as_list()
  shape[0] = n
  u = tf.Variable(tf.zeros(shape=shape, dtype=y.dtype))

  # Define the action of the transposed Jacobian.
  JTus = tf.gradients(tf.reduce_sum(y*u), xs)

  # Define the action of the Jacobian.
  JTus_dot_vs = [tf.reduce_sum(JTu*v) for JTu, v in zip(JTus, vs)]
  Jv = tf.gradients(JTus_dot_vs, u)[0]

  # Define the action of the Hessian on Jv.
  grad_l = tf.gradients(l, y)[0]
  HJv = tf.gradients(tf.reduce_sum(grad_l*Jv), y, stop_gradients=Jv)

  # Define the action of the Gauss-Newton operator.
  JTHJvs = tf.gradients(tf.reduce_sum(y*HJv), xs, stop_gradients=HJv)

  return JTHJvs
