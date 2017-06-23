# coding=utf-8

# Network in Network CIFAR10 Model
# Original source: https://gist.github.com/mavenlin/e56253735ef32c3c296d
# License: unknown

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/cifar10/model.pkl

"""
IMPORTANT: This net does not flip filters in convolution layers.
Make sure the converted spiking net behaves the same.

"""


def build_network():
    """Build network.

    Returns
    -------

    """

    import lasagne
    import theano
    import theano.tensor as t
    from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.layers import Pool2DLayer as PoolLayer

    # Prepare Theano variables for inputs and targets
    input_var = t.tensor4('inputs')
    target = t.matrix('targets')

    cnn = {'input': InputLayer(shape=(None, 3, 32, 32), input_var=input_var)}
    cnn['conv1'] = ConvLayer(cnn['input'],
                             num_filters=192,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    cnn['cccp1'] = ConvLayer(
        cnn['conv1'], num_filters=160, filter_size=1, flip_filters=False)
    cnn['cccp2'] = ConvLayer(
        cnn['cccp1'], num_filters=96, filter_size=1, flip_filters=False)
    cnn['pool1'] = PoolLayer(cnn['cccp2'],
                             pool_size=3,
                             stride=2,
                             mode='max',
                             ignore_border=False)
    cnn['drop3'] = DropoutLayer(cnn['pool1'], p=0.5)
    cnn['conv2'] = ConvLayer(cnn['drop3'],
                             num_filters=192,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    cnn['cccp3'] = ConvLayer(
        cnn['conv2'], num_filters=192, filter_size=1, flip_filters=False)
    cnn['cccp4'] = ConvLayer(
        cnn['cccp3'], num_filters=192, filter_size=1, flip_filters=False)
    cnn['pool2'] = PoolLayer(cnn['cccp4'],
                             pool_size=3,
                             stride=2,
                             mode='average_exc_pad',
                             ignore_border=False)
    cnn['drop6'] = DropoutLayer(cnn['pool2'], p=0.5)
    cnn['conv3'] = ConvLayer(cnn['drop6'],
                             num_filters=192,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    cnn['cccp5'] = ConvLayer(
        cnn['conv3'], num_filters=192, filter_size=1, flip_filters=False)
    cnn['cccp6'] = ConvLayer(
        cnn['cccp5'], num_filters=10, filter_size=1, flip_filters=False)
    cnn['pool3'] = PoolLayer(cnn['cccp6'],
                             pool_size=8, stride=1,
                             mode='average_exc_pad',
                             ignore_border=False)
    cnn['output'] = FlattenLayer(cnn['pool3'])

    test_output = lasagne.layers.get_output(cnn['output'], deterministic=True)
    test_loss = t.mean(t.sqr(t.maximum(0., 1. - target * test_output)))
    test_err = t.mean(t.neq(t.argmax(test_output, axis=1),
                            t.argmax(target, axis=1)),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target], [test_loss, test_err])

    return {'model': cnn['output'], 'val_fn': val_fn}
