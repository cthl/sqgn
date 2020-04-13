# Neural network training with a stochastic quasi-Gauss-Newton method

This repository contains the implementation of the stochastic quasi-Gauss-Newton method presented and evaluated in our paper entitled *Deep Neural Network Learning with Second-Order Optimizers - a Practical Study with a Stochastic Quasi-Gauss-Newton Method*, available on [ArXiv](https://arxiv.org/abs/2004.03040).

The `sqgn` directory contains a Python module that implements SQGN and other
optimization methods for neural network training (SGD, Adam, etc.).

The `mnist_tf` directory contains a TensorFlow-based neural network for the
MNIST dataset. The network is trained using optimizers from the `sqgn` module.
This example demonstrates how the different optimizers can be called, and it can
be used to compare their convergence and computational performance. The script
`mnist_tf.py` supports the following command line arguments:

*  `-num_data`

    Number of samples to be used for training
    (default: 60000, the entire MNIST training dataset)

*  `-num_epochs`

    Number of training epochs
    (default: 100)

*  `-batch_size`

    Mini-batch size for optimizers
    (default: 1000)    

*  `-batch_size_hess`

    Mini-batch size for Hessian evaluations and evaluations of the Gauss-Newton
    operator
    (default: 1000)

*  `-opt_name`

    The optimizer to be used for training. Possible values include 'sqgn', sgd',
    'adam', 'lbfgs', 'newtoncg', 'gaussnewtoncg', and 'newtontr'.
    (default: adam)

*  `-lr`

    Learning rate
    (default: 1.0e-2)

*  `-reg`

    Regularization parameter
    (default: 0.1)

*  `-hist`

    The maximum number of curvature pairs for quasi-Newton and
    quasi-Gauss-Newton methods
    (default: 20)

*  `-grad_agg`

    The type of gradient aggregation to be used. Possible options are raw
    gradients ('raw_grad'), i.e., no aggregation, and stochastic
    variance-reduced gradients ('svrg').
    (default: raw_grad)

*  `-grad_agg_interval`

    Number of mini-batches between evaluations of the full gradient for
    stochastic variance-reduced gradients
    (default: 10)

*  `sqn_update_interval`

    Number of mini-batches between updates of the approximation to the Hessian
    or the Gauss-Newton operator
    (default: 1)

All code is based on Python 3 and TensorFlow 1.6.0.
