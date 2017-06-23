# coding=utf-8

"""LeNet on MNIST with lasagne."""

import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, BatchNormLayer, DenseLayer
from lasagne.layers import NonlinearityLayer, MaxPool2DLayer, DropoutLayer
from keras.datasets import mnist as dataset
from keras.utils.np_utils import to_categorical
import theano.tensor as t
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')

batch_size = 128
nb_classes = 10
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 28, 28
chnls = 1

(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train = x_train.reshape(x_train.shape[0], chnls, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], chnls, img_rows, img_cols)
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
x_train /= 255
x_test /= 255


def build_network():
    """Build LeNet"""

    input_var = t.tensor4('inputs')
    target = t.matrix('targets')

    cnn = InputLayer((None, 1, 28, 28), input_var)
    cnn = Conv2DLayer(cnn, 16, 5, 2)
    cnn = BatchNormLayer(cnn)
    cnn = NonlinearityLayer(cnn, lasagne.nonlinearities.rectify)

    cnn = Conv2DLayer(cnn, 32, (3, 3), (1, 1))
    cnn = BatchNormLayer(cnn)
    cnn = NonlinearityLayer(cnn, lasagne.nonlinearities.rectify)
    cnn = MaxPool2DLayer(cnn, (2, 2))
    cnn = DropoutLayer(cnn, 0.1)

    cnn = DenseLayer(cnn, 800)
    cnn = BatchNormLayer(cnn)
    cnn = NonlinearityLayer(cnn, lasagne.nonlinearities.rectify)
    cnn = DropoutLayer(cnn, 0.1)

    cnn = DenseLayer(cnn, 10, nonlinearity=lasagne.nonlinearities.softmax)

    train_out = lasagne.layers.get_output(cnn, deterministic=False)
    train_loss = lasagne.objectives.categorical_crossentropy(train_out, target)
    train_loss = lasagne.objectives.aggregate(train_loss, mode='mean')

    parameters = lasagne.layers.get_all_params(cnn, trainable=True)
    updates = lasagne.updates.adam(train_loss, parameters)
    train_fn = theano.function([input_var, target], train_loss, updates=updates)

    test_out = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_out, target)
    test_loss = lasagne.objectives.aggregate(test_loss, mode='mean')
    err = t.mean(t.neq(t.argmax(test_out, 1), t.argmax(target, 1)),
                 dtype=theano.config.floatX)
    val_fn = theano.function([input_var, target], [test_loss, err])

    return cnn, train_fn, val_fn


def train_epoch(x, y, train_fn, batchsize):
    """Train batch-wise.

    Parameters
    ----------

    x: np.array
        Input samples.
    y: np.array
        Target classes.
    train_fn: theano.function
        Theano function for training.
    batchsize: int
        Batch size.

    Returns
    -------

    : float
        The training loss.
    """

    batches = int(len(x) / batchsize)
    loss = 0
    for i in range(batches):
        loss += train_fn(x[i * batchsize:(i + 1) * batchsize],
                         y[i * batchsize:(i + 1) * batchsize])
    return loss / batches


def test_epoch(x, y, test_fn, batchsize):
    """Test batch-wise.

    Parameters
    ----------

    x: np.array
        Input samples.
    y: np.array
        Target classes.
    test_fn: theano.function
        Theano function for training.
    batchsize: int
        Batch size.

    Returns
    -------

    err: float
        The testing error.
    loss: float
        The testing loss.
    """

    err = 0
    loss = 0
    batches = int(len(x) / batch_size)
    for i in range(batches):
        new_loss, new_err = test_fn(x[i * batchsize:(i + 1) * batchsize],
                                    y[i * batch_size:(i + 1) * batchsize])
        err += new_err
        loss += new_loss
    return err / batches, loss / batches


def shuffle(x, y):
    """Shuffle dataset.

    Parameters
    ----------

    x: np.array
        Input samples.
    y: np.array
        Input classes.

    Returns
    -------

    x: np.array
        Shuffled samples.
    y: np.array
        Shuffled classes.
    """

    import numpy as np

    shuffle_parts = 1
    chunk_size = int(len(x) / shuffle_parts)
    shuffled_range = list(range(chunk_size))

    x_buffer = np.copy(x[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):
        np.random.shuffle(shuffled_range)

        for i in range(chunk_size):
            x_buffer[i] = x[k * chunk_size + shuffled_range[i]]
            y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

        x[k * chunk_size:(k + 1) * chunk_size] = x_buffer
        y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

    return x, y


if __name__ == "__main__":
    from snntoolbox.datasets.utils import save_parameters

    model, train_func, test_func = build_network()

    print("Training...")
    for epoch in range(nb_epoch):
        print("Epoch: {}".format(epoch))
        print("Training loss: {}".format(train_epoch(x_train, y_train, train_func,
                                            batch_size)))
        x_train, y_train = shuffle(x_train, y_train)

        val_err, val_loss = test_epoch(x_test, y_test, test_func, batch_size)
        print("Validation accuracy: {}".format((1-val_err)*100))

    score = test_func(x_test, y_test)

    params = lasagne.layers.get_all_param_values(model)
    save_parameters(params, '{:2.2f}.h5'.format((1-score[1])*100))
