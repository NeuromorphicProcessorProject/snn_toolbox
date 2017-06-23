# coding=utf-8

"""BinaryNet"""

from __future__ import absolute_import
from __future__ import print_function

import lasagne

from scripts.ann_architectures.BinaryConnect import binary_net


def build_network():
    """Build BinaryNet.

    Returns
    -------

    : tuple[lasagne.layers.Layer, theano.function, theano.function]
        The network (``model``), and functions to train and test
        (``train_func``, ``val_fn``).
    """

    import theano
    import theano.tensor as t
    from collections import OrderedDict

    # BN parameters
    # alpha is the exponential moving average factor
    alpha = .1
    epsilon = 1e-4

    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation: {}".format(activation))

    # BinaryConnect
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-h,+h) are the two binary values
    # h = "Glorot"
    h = 1.
    print("h = "+str(h))
    # w_lr_scale = 1.
    # "Glorot" means we are using the coefficients from Glorot's paper
    w_lr_scale = "Glorot"
    print("w_lr_scale = "+str(w_lr_scale))

    # Prepare Theano variables for inputs and targets
    input_var = t.tensor4('inputs')
    target = t.matrix('targets')
    lr = t.scalar('lr', dtype=theano.config.floatX)

    cnn = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                    input_var=input_var)

    cnn = binary_net.Conv2DLayer(cnn, binary=binary, stochastic=stochastic,
                                 H=h, W_LR_scale=w_lr_scale, num_filters=32,
                                 filter_size=(3, 3), nonlinearity=activation)

    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation)

    cnn = binary_net.Conv2DLayer(cnn, binary=binary, stochastic=stochastic,
                                 H=h, W_LR_scale=w_lr_scale, num_filters=32,
                                 filter_size=(3, 3), nonlinearity=activation)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.DropoutLayer(cnn, 0.25)

    cnn = binary_net.DenseLayer(cnn, binary=binary, stochastic=stochastic, H=h,
                                W_LR_scale=w_lr_scale, num_units=128,
                                nonlinearity=activation)

    cnn = lasagne.layers.DropoutLayer(cnn)

    cnn = binary_net.DenseLayer(cnn, binary=binary, stochastic=stochastic, H=h,
                                W_LR_scale=w_lr_scale, num_units=10,
                                nonlinearity=lasagne.nonlinearities.softmax)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)

    # squared hinge loss
    loss = lasagne.objectives.categorical_crossentropy(train_output, target)
    loss = lasagne.objectives.aggregate(loss, mode='mean')

    if binary:
        from itertools import chain
        # w updates
        w = lasagne.layers.get_all_params(cnn, binary=True)
        w_grads = binary_net.compute_grads(loss, cnn)
        updates = lasagne.updates.adam(loss_or_grads=w_grads, params=w,
                                       learning_rate=lr)
        updates = binary_net.clipping_scaling(updates, cnn)

        # other parameters updates
        parameters = lasagne.layers.get_all_params(cnn, trainable=True,
                                                   binary=False)
        updates = OrderedDict(chain(updates.items(), lasagne.updates.adam(
            loss_or_grads=loss, params=parameters, learning_rate=lr).items()))

    else:
        parameters = lasagne.layers.get_all_params(cnn, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=parameters,
                                       learning_rate=lr)

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_output,
                                                            target)
    test_loss = lasagne.objectives.aggregate(test_loss, mode='mean')
    test_err = t.mean(t.neq(t.argmax(test_output, axis=1),
                            t.argmax(target, axis=1)),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_func = theano.function([input_var, target, lr], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_func = theano.function([input_var, target], [test_loss, test_err])

    return cnn, train_func, val_func


if __name__ == "__main__":

    import numpy as np
    from snntoolbox.datasets.utils import load_npz, save_parameters
    np.random.seed(1234)

    # Training parameters
    batch_size = 128
    print("batch_size = "+str(batch_size))
    num_epochs = 10
    print("num_epochs = "+str(num_epochs))

    # Decaying LR
    LR_start = 0.001
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))

    print("Loading dataset...")

    path_to_dataset = '/home/rbodo/.snntoolbox/Datasets/mnist/cnn'
    X_train = load_npz(path_to_dataset, 'x_norm.npz').astype('float32')
    Y_train = load_npz(path_to_dataset, 'y_train.npz').astype('float32')
    X_test = load_npz(path_to_dataset, 'x_test.npz').astype('float32')
    Y_test = load_npz(path_to_dataset, 'y_test.npz').astype('float32')

    print('Building the CNN...')

    model, train_fn, val_fn = build_network()

    print('Training...')

    binary_net.train(train_fn, val_fn, model, batch_size, LR_start, LR_decay,
                     num_epochs, X_train, Y_train, X_test, Y_test, X_test,
                     Y_test, shuffle_parts=1)

    W = lasagne.layers.get_all_layers(model)[1].W.get_value()

    import matplotlib.pyplot as plt
    plt.hist(W.flatten())
    plt.title("Weight distribution of first hidden convolution layer")

    # Dump the network weights to a file
    filepath = 'binarynet_mnist.h5'
    params = lasagne.layers.get_all_param_values(model)
    save_parameters(params, filepath)
