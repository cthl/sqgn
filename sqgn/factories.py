"""
factories.py - Factory functions for optimizers.

Author: Christopher Thiele
"""

from sqgn.sgd import SGD
from sqgn.rmsprop import RMSprop
from sqgn.adam import Adam
from sqgn.lbfgs import LBFGS, SQN
from sqgn.newton import NewtonCG, GaussNewtonCG, NewtonTR, CurveBall

from sqgn.line_search import FullStep, Armijo

from sqgn.gradient_aggregator import RawGradient, SVRG

def create_optimizer(name, loss, weights, conf=None, sess=None):
  """Create an optimizer based on its name."""
  if name == 'sgd':
    return SGD(loss, weights, conf, sess)
  elif name == 'rmsprop':
    return RMSprop(loss, weights, conf, sess)
  elif name == 'adam':
    return Adam(loss, weights, conf, sess)
  elif name == 'lbfgs':
    return LBFGS(loss, weights, conf, sess)
  elif name == 'sqn':
    return SQN(loss, weights, conf, sess)
  elif name == 'newtoncg':
    return NewtonCG(loss, weights, conf, sess)
  elif name == 'gaussnewtoncg':
    return GaussNewtonCG(loss, weights, conf, sess)
  elif name == 'newtontr':
    return NewtonTR(loss, weights, conf, sess)
  elif name == 'curveball':
    return CurveBall(loss, weights, conf, sess)
  else:
    assert False, 'Invalid optimizer name!'

def create_line_search(name, loss, weights, dws, grads, conf=None, sess=None):
  """Create a line search method based on its name."""
  if name == 'full_step':
    return FullStep(loss, weights, dws, grads, conf, sess)
  elif name == 'armijo':
    return Armijo(loss, weights, dws, grads, conf, sess)
  else:
    assert False, 'Invalid line search name!'

def create_gradient_aggregator(name, loss, weights, sess=None):
  """Create a gradient aggregator based on its name."""
  if name == 'raw_grad':
    return RawGradient(loss, weights, sess)
  elif name == 'svrg':
    return SVRG(loss, weights, sess)
  else:
    assert False, 'Invalid gradient aggregator name!'
