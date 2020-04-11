"""
sqgn

This package contains implementations of different optimizers for
TensorFlow-based machine learning applications.

New users should start by reviewing the documentation of the abstract
'Optimizer' class.
It describes the interface that all optimizers in this package have in common.
"""

from sqgn.sgd import SGD
from sqgn.rmsprop import RMSprop
from sqgn.adam import Adam
from sqgn.lbfgs import LBFGS, SQN
from sqgn.newton import NewtonCG, GaussNewtonCG, NewtonTR, CurveBall

from sqgn.krylov import CG, TRCG

from sqgn.line_search import FullStep, Armijo

from sqgn.gauss_newton import JTJv, JTHJv

from sqgn.gradient_aggregator import RawGradient, SVRG

from sqgn.factories import create_optimizer, \
                           create_line_search, \
                           create_gradient_aggregator
