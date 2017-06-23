# coding=utf-8

"""

This python script trains a ConvNet on CIFAR-10 with BinaryNet.

It should run for about 15 hours on a GeForce GTX 980 Ti GPU.

The final test error should be around 11.40%.

Source:

https://github.com/MatthieuCourbariaux/BinaryNet

"""


from __future__ import print_function

import lasagne
# specifying the gpu to use
import theano.sandbox.cuda

from scripts.ann_architectures.BinaryConnect import binary_net

theano.sandbox.cuda.use('gpu0')


def build_network():
    """Build network.

    Returns
    -------

    """

    import theano
    import theano.tensor as t
    from collections import OrderedDict

    # BN parameters
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))

    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")

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

    cnn = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                    input_var=input_var)

#    #Experimental: Train on binarized input!
#    model = lasagne.layers.NonlinearityLayer(
#            model,
#            nonlinearity=activation)

    # 128C3-128C3-P2
    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=h,
            W_LR_scale=w_lr_scale,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=h,
            W_LR_scale=w_lr_scale,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    # 256C3-256C3-P2
    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=h,
            W_LR_scale=w_lr_scale,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=h,
            W_LR_scale=w_lr_scale,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    # 512C3-512C3-P2
    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=h,
            W_LR_scale=w_lr_scale,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = binary_net.Conv2DLayer(
            cnn,
            binary=binary,
            stochastic=stochastic,
            H=h,
            W_LR_scale=w_lr_scale,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    # print(model.output_shape)

    # 1024FP-1024FP-10FP
    cnn = binary_net.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=h,
                W_LR_scale=w_lr_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = binary_net.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=h,
                W_LR_scale=w_lr_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = binary_net.DenseLayer(
                cnn,
                binary=binary,
                stochastic=stochastic,
                H=h,
                W_LR_scale=w_lr_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)

    # squared hinge loss
    loss = t.mean(t.sqr(t.maximum(0., 1.-target*train_output)))

    if binary:
        from itertools import chain
        # w updates
        w = lasagne.layers.get_all_params(cnn, binary=True)
        w_grads = binary_net.compute_grads(loss, cnn)
        updates = lasagne.updates.adam(loss_or_grads=w_grads, params=w,
                                       learning_rate=lr)
        updates = binary_net.clipping_scaling(updates, cnn)

        # other parameters updates
        params = lasagne.layers.get_all_params(cnn, trainable=True,
                                               binary=False)
        updates = OrderedDict(chain(updates.items(), lasagne.updates.adam(
            loss_or_grads=loss, params=params, learning_rate=lr).items()))

    else:
        params = lasagne.layers.get_all_params(cnn, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params,
                                       learning_rate=lr)

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = t.mean(t.sqr(t.maximum(0., 1.-target*test_output)))
    test_err = t.mean(t.neq(t.argmax(test_output, axis=1),
                            t.argmax(target, axis=1)),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target, lr], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target], [test_loss, test_err])

    return cnn, train_fn, val_fn


if __name__ == "__main__":

    from pylearn2.datasets.cifar10 import CIFAR10
    import numpy as np
    from snntoolbox.datasets.utils import save_parameters
#    from snntoolbox.model_libs.lasagne_input_lib import load_parameters
    np.random.seed(1234)  # for reproducibility?

    # Training parameters
    batch_size = 50
    print("batch_size = "+str(batch_size))
    num_epochs = 500
    print("num_epochs = "+str(num_epochs))

    # Decaying LR
    LR_start = 0.001
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...

    train_set_size = 45000
    print("train_set_size = "+str(train_set_size))
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))

    print('Loading CIFAR-10 dataset...')

    train_set = CIFAR10(which_set="train", start=0, stop=train_set_size)
    valid_set = CIFAR10(which_set="train", start=train_set_size, stop=50000)
    test_set = CIFAR10(which_set="test")

    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    train_set.X = np.reshape(
        np.subtract(np.multiply(2./255., train_set.X), 1.), (-1, 3, 32, 32))
    valid_set.X = np.reshape(
        np.subtract(np.multiply(2./255., valid_set.X), 1.), (-1, 3, 32, 32))
    test_set.X = np.reshape(
        np.subtract(np.multiply(2./255., test_set.X), 1.), (-1, 3, 32, 32))

    # flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)

    # Onehot the targets
    train_set.y = np.float32(np.eye(10)[train_set.y])
    valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])

    # for hinge loss
    train_set.y = 2 * train_set.y - 1.
    valid_set.y = 2 * valid_set.y - 1.
    test_set.y = 2 * test_set.y - 1.

    print('Building the CNN...')

    model, train_func, val_func = build_network()

    # Experimental: Initialize with pretrained weights, refine with binarized
    #  input.
#    params = load_parameters(
#        '/home/rbodo/.snntoolbox/data/cifar10/88.63/INI/88.63.h5')
#    lasagne.layers.set_all_param_values(model, params)

    print('Training...')

    binary_net.train(train_func, val_func, model, batch_size, LR_start,
                     LR_decay, num_epochs, train_set.X, train_set.y,
                     valid_set.X, valid_set.y, test_set.X, test_set.y,
                     shuffle_parts=shuffle_parts)

    W = lasagne.layers.get_all_layers(model)[1].W.get_value()

    import matplotlib.pyplot as plt
    plt.hist(W.flatten())
    plt.title("Weight distribution of first hidden convolution layer")

    # Dump the network weights to a file
    filepath = '70.14.h5'
    parameters = lasagne.layers.get_all_param_values(model)
    save_parameters(parameters, filepath)
