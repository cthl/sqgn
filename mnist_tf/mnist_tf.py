#! /usr/bin/env python3

import random
import time
import argparse
from functools import reduce
import sys

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)
import numpy as np
import scipy.io

sys.path.append('../sqgn')
import sqgn

# Parse command line arguments.
parser = argparse.ArgumentParser(description='mnist_tf')
parser.add_argument('-num_data', type=int, default=60000)
parser.add_argument('-num_epochs', type=int, default=100)
parser.add_argument('-batch_size', type=int, default=1000)
parser.add_argument('-batch_size_hess', type=int, default=1000)
parser.add_argument('-seed', type=int, default=None)
parser.add_argument('-opt_name', default='adam')
parser.add_argument('-lr', type=float, default=1.0e-2)
parser.add_argument('-reg', type=float, default=0.1)
parser.add_argument('-hist', type=int, default=20)
parser.add_argument('-grad_agg', default='raw_grad')
parser.add_argument('-grad_agg_interval', type=int, default=10)
parser.add_argument('-sqn_update_interval', type=int, default=1)
parser.add_argument('-eps', type=float, default=1.0e-7)
args = parser.parse_args()

# Initialize random number generator.
seed = args.seed
if seed is None:
  print('Using default random seed (system time).')
else:
  print('Using FIXED random seed of %d.' % seed)
random.seed(seed)

# Convert big-endian to little-endian integers.
def swap32(x):
    return (((x << 24) & 0xFF000000) |
            ((x <<  8) & 0x00FF0000) |
            ((x >>  8) & 0x0000FF00) |
            ((x >> 24) & 0x000000FF))

# Load an MNIST data set.
def mnist_load(filename, num_data=-1):
  f = open(filename, 'rb')
  data = np.fromfile(f, dtype=np.int32, count=2)
  # Get MNIST magic number and number of images/labels.
  magic = swap32(data[0])
  num_data_max = swap32(data[1])
  assert num_data <= num_data_max, 'Requested more data than is available!'

  if magic == 2051:
    # File contains images.
    data = np.fromfile(f, dtype=np.int32, count=2)
    height = swap32(data[0])
    width = swap32(data[1])
  elif magic == 2049:
    # File contains labels.
    height = 1
    width = 1
  else:
    assert False, 'Unknown MNIST magic number!'

  # Load data and reshape to correct dimensions.
  data = np.fromfile(f, dtype=np.uint8, count=num_data*height*width)
  data = np.reshape(data, (num_data, height, width))

  return data

# Load MNIST data into NumPy arrays.
num_data = args.num_data
train_data = mnist_load('train-images-idx3-ubyte', num_data)
train_labels_raw = mnist_load('train-labels-idx1-ubyte', num_data)
test_data = mnist_load('t10k-images-idx3-ubyte')
test_labels_raw = mnist_load('t10k-labels-idx1-ubyte')

# Convert the labels into the correct format.
train_labels = np.zeros((train_labels_raw.shape[0], 10), dtype=np.float32)
for i in range(train_labels_raw.shape[0]):
  label = train_labels_raw[i]
  train_labels[i][label] = 1.0
test_labels = np.zeros((test_labels_raw.shape[0], 10), dtype=np.float32)
for i in range(test_labels_raw.shape[0]):
  label = test_labels_raw[i]
  test_labels[i][label] = 1.0

# Create TensorFlow session.
sess = tf.Session()

# Construct the neural network.
x = tf.placeholder(tf.float32, [None, 28, 28], name='x')
y_ref = tf.placeholder(tf.float32, [None, 10], name='y_ref')

glorot = tf.glorot_uniform_initializer(dtype=tf.float32)

y = tf.reshape(x, [-1, 28, 28, 1])

shape0 = [4, 4, 1, 2]
W0 = tf.Variable(glorot(shape0))
y = tf.nn.conv2d(y, W0, [1, 1, 1, 1], 'SAME')
y = tf.nn.relu(y)
y = tf.nn.max_pool(y, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
print(y.shape)

shape1 = [4, 4, 2, 4]
W1 = tf.Variable(glorot(shape1))
y = tf.nn.conv2d(y, W1, [1, 1, 1, 1], 'SAME')
y = tf.nn.relu(y)
y = tf.nn.max_pool(y, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
print(y.shape)

shape2 = [4, 4, 4, 8]
W2 = tf.Variable(glorot(shape2))
y = tf.nn.conv2d(y, W2, [1, 1, 1, 1], 'SAME')
y = tf.nn.relu(y)
y = tf.nn.max_pool(y, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
print(y.shape)

y = tf.reshape(y, [-1, 128])
shape_fc = [128, 10]
Wfc = tf.Variable(glorot(shape_fc))
bfc = tf.Variable(tf.zeros([10]))
y = tf.matmul(y, Wfc) + bfc

# Define loss and weights.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_ref, logits=y))
weights = tf.trainable_variables()
for w in weights:
  print(w.shape.as_list())
for w in weights:
  print(w)

# Configure the training procedure.
num_epochs = args.num_epochs
ensure_coverage = True
hessian_sub_batch = True
batch_size = args.batch_size
batch_size_hess = args.batch_size_hess
if ensure_coverage:
  assert num_data%batch_size == 0, \
         'Data set size is not divisible by the batch size!'
if hessian_sub_batch:
  assert batch_size >= batch_size_hess, \
         'Batch size for Hessian evaluations is too large!'
else:
  assert not ensure_coverage, \
         'Cannot ensure sample coverage when Hessian is sampled independently!'
num_batches = num_data//batch_size

# Create the optimizer.
opt_name = args.opt_name
if opt_name == 'sgd':
  conf = sqgn.SGD.get_default_conf()
  conf['lr'] = args.lr
  conf['grad_agg'] = args.grad_agg
elif opt_name == 'adam':
  conf = sqgn.Adam.get_default_conf()
  conf['lr'] = args.lr
elif opt_name == 'lbfgs':
  conf = sqgn.LBFGS.get_default_conf()
  conf['lr'] = args.lr
  conf['reg'] = args.reg
  conf['hist'] = args.hist
  conf['neg_curv_update'] = 'reset'
elif opt_name == 'sqn' or opt_name == 'sqgn':
  conf = sqgn.SQN.get_default_conf()
  conf['lr'] = args.lr
  conf['reg'] = args.reg
  conf['hist'] = args.hist
  conf['eps'] = args.eps
  # Use stochastic quasi-Gauss-Newton, not stochastic quasi-Newton.
  conf['gauss_newton'] = True
  conf['num_samples'] = batch_size_hess
  conf['pred'] = y
  conf['grad_agg'] = args.grad_agg
  conf['update_interval'] = args.sqn_update_interval
elif opt_name == 'newtoncg':
  conf = sqgn.NewtonCG.get_default_conf()
  conf['line_search'] = 'full_step'
  conf['lr'] = args.lr
  conf['reg'] = args.reg
  #conf['line_search'] = 'armijo'
  #line_search_conf = sqgn.Armijo.get_default_conf()
  #conf['line_search_conf'] = line_search_conf
elif opt_name == 'gaussnewtoncg':
  conf = sqgn.GaussNewtonCG.get_default_conf()
  conf['num_samples'] = batch_size_hess
  conf['pred'] = y
  conf['reg'] = args.reg
  conf['lr'] = args.lr
elif opt_name == 'newtontr':
  conf = sqgn.NewtonTR.get_default_conf()
else:
  assert False, 'Invalid optimizer name!'
print(opt_name)
print(conf)
opt = sqgn.create_optimizer(opt_name, loss, weights, conf=conf, sess=sess)

# Initialize TensorFlow variables.
sess.run(tf.global_variables_initializer())

def get_batch(idxs, batch_size, batch_size_hess=0, track_coverage=True):
  batch_idxs = []
  if track_coverage:
    for _ in range(batch_size):
      i = random.randrange(len(idxs))
      batch_idxs.append(idxs[i])
      del idxs[i]
  else:
    for _ in range(batch_size):
      i = random.randrange(num_data)
      batch_idxs.append(i)

  x_batch = train_data[batch_idxs]
  y_ref_batch = train_labels[batch_idxs]

  if batch_size_hess == 0:
    return x_batch, y_ref_batch

  # Create a smaller batch for Hessian evaluations.
  batch_idxs_hess = []
  for _ in range(batch_size_hess):
    i = random.randrange(len(batch_idxs))
    batch_idxs_hess.append(batch_idxs[i])
    del batch_idxs[i]

  x_batch_hess = train_data[batch_idxs_hess]
  y_ref_batch_hess = train_labels[batch_idxs_hess]

  return x_batch, y_ref_batch, x_batch_hess, y_ref_batch_hess

# Initialize time measurement.
t_opt_total = 0.0
t_var_red = 0.0

# Training loop
for epoch in range(num_epochs):
  print('START of epoch %d/%d' % (epoch + 1, num_epochs))

  # A new epoch begins, so all data can be used.
  idxs = list(range(num_data))

  for batch in range(num_batches):
    print(' Batch %d/%d: ' % (batch + 1, num_batches), end='')

    # Update reference gradient if necessary.
    if opt_name == 'sgd' or opt_name == 'sqn':
      var_red = True
    else:
      var_red = False
    if var_red and batch%args.grad_agg_interval == 0:
      grad_agg = opt.get_gradient_aggregator()
      fdict = {x: train_data, y_ref: train_labels}

      t_opt_start = time.time()
      grad_agg.update_reference(fdict)
      if epoch > 0:
        t_var_red += time.time() - t_opt_start
        t_opt_total += time.time() - t_opt_start

    # Prepare batches.
    if hessian_sub_batch:
      x_batch, y_ref_batch, x_batch_hess, y_ref_batch_hess \
        = get_batch(idxs,
                    batch_size,
                    batch_size_hess,
                    track_coverage=ensure_coverage)
    else:
      x_batch, y_ref_batch = get_batch(idxs,
                                       batch_size,
                                       batch_size_hess=0,
                                       track_coverage=False)
      x_batch_hess, y_ref_batch_hess = get_batch(idxs,
                                                 batch_size_hess,
                                                 batch_size_hess=0,
                                                 track_coverage=False)

    # Create feed dictionaries.
    fdict = {x: x_batch, y_ref: y_ref_batch}
    fdict_hess = {x: x_batch_hess, y_ref: y_ref_batch_hess}

    # Call optimizer.
    t_opt_start = time.time()
    opt.minimize(fdict, samples_hessian=fdict_hess)
    if epoch > 0:
      t_opt_total += time.time() - t_opt_start

    batch_loss = loss.eval(session=sess, feed_dict=fdict)
    print('loss = %e' % batch_loss, end='\r')

  # Compute and report loss and accuracy.
  fdict = {x: test_data, y_ref: test_labels}
  test_loss = loss.eval(session=sess, feed_dict=fdict)

  y_vals = y.eval(session=sess, feed_dict=fdict)
  correct = 0
  for i in range(test_labels.shape[0]):
    label = -1
    prob = -1.0
    for j in range(10):
      if y_vals[i][j] > prob:
        label = j
        prob = y_vals[i][j]
    if label == test_labels_raw[i]:
      correct += 1
  test_acc = 100.0*correct/test_labels.shape[0]

  print('END of epoch %d/%d' % (epoch + 1, num_epochs))
  print(' -> test loss = %e' % test_loss)
  print(' -> acc. = %.1f%%' % test_acc)
  print(' -> elapsed time (opt.) = %.1f s' % t_opt_total)
  if epoch > 0:
    print(' -> avg. time per epoch (svrg) = %.3f s' % (t_var_red/epoch))
    print(' -> avg. time per epoch (opt.) = %.3f s' % (t_opt_total/(epoch)))
    print(' -> avg. time per batch (opt.) = %.3f s' \
          % (t_opt_total/((epoch)*num_batches)))
