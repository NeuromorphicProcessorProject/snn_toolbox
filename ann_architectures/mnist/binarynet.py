from __future__ import absolute_import
from __future__ import print_function

import lasagne
from ann_architectures.BinaryConnect import binary_net
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')


def build_network():

    import theano
    import theano.tensor as T
    from collections import OrderedDict

    # BN parameters
    # alpha is the exponential moving average factor
    alpha = .1
    epsilon = 1e-4

    # BinaryOut
    activation = binary_net.binary_sigmoid_unit
    print("activation: {}".format(activation))

    # BinaryConnect
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.
    # "Glorot" means we are using the coefficients from Glorot's paper
    W_LR_scale = "Glorot"
    print("W_LR_scale = "+str(W_LR_scale))

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                    input_var=input_var)

    cnn = binary_net.Conv2DLayer(cnn, binary=binary, stochastic=stochastic,
                                 H=H, W_LR_scale=W_LR_scale, num_filters=32,
                                 filter_size=(3, 3), nonlinearity=activation)

    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation)

    cnn = binary_net.Conv2DLayer(cnn, binary=binary, stochastic=stochastic,
                                 H=H, W_LR_scale=W_LR_scale, num_filters=32,
                                 filter_size=(3, 3), nonlinearity=activation)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.DropoutLayer(cnn, 0.25)

    cnn = binary_net.DenseLayer(cnn, binary=binary, stochastic=stochastic, H=H,
                                W_LR_scale=W_LR_scale, num_units=128,
                                nonlinearity=activation)

    cnn = lasagne.layers.DropoutLayer(cnn)

    cnn = binary_net.DenseLayer(cnn, binary=binary, stochastic=stochastic, H=H,
                                W_LR_scale=W_LR_scale, num_units=10,
                                nonlinearity=lasagne.nonlinearities.softmax)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)

    # squared hinge loss
    loss = lasagne.objectives.categorical_crossentropy(train_output, target)
    loss = lasagne.objectives.aggregate(loss, mode='mean')

    if binary:
        from itertools import chain
        # W updates
        W = lasagne.layers.get_all_params(cnn, binary=True)
        W_grads = binary_net.compute_grads(loss, cnn)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W,
                                       learning_rate=LR)
        updates = binary_net.clipping_scaling(updates, cnn)

        # other parameters updates
        params = lasagne.layers.get_all_params(cnn, trainable=True,
                                               binary=False)
        updates = OrderedDict(chain(updates.items(), lasagne.updates.adam(
            loss_or_grads=loss, params=params, learning_rate=LR).items()))

    else:
        params = lasagne.layers.get_all_params(cnn, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params,
                                       learning_rate=LR)

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_output,
                                                            target)
    test_loss = lasagne.objectives.aggregate(test_loss, mode='mean')
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1),
                            T.argmax(target, axis=1)),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target], [test_loss, test_err])

    return cnn, train_fn, val_fn


if __name__ == "__main__":

    import numpy as np
    from snntoolbox.io_utils.common import load_dataset, save_parameters
    np.random.seed(1234)

    # Training parameters
    batch_size = 128
    print("batch_size = "+str(batch_size))
    num_epochs = 50
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
    X_train = load_dataset(path_to_dataset, 'X_norm.npz')
    Y_train = load_dataset(path_to_dataset, 'Y_train.npz')
    X_test = load_dataset(path_to_dataset, 'X_test.npz')
    Y_test = load_dataset(path_to_dataset, 'Y_test.npz')

    print('Building the CNN...')

    cnn, train_fn, val_fn = build_network()

    print('Training...')

    binary_net.train(train_fn, val_fn, cnn, batch_size, LR_start, LR_decay,
                     num_epochs, X_train, Y_train, X_test, Y_test, X_test,
                     Y_test, shuffle_parts=1)

    W = lasagne.layers.get_all_layers(cnn)[1].W.get_value()

    import matplotlib.pyplot as plt
    plt.hist(W.flatten())
    plt.title("Weight distribution of first hidden convolution layer")

    # Dump the network weights to a file
    filepath = '70.14.h5'
    params = lasagne.layers.get_all_param_values(cnn)
    save_parameters(params, filepath)
