# coding=utf-8

"""
Refine pretrained inception-v3, with Average instead of MaxPooling.

Inception-v3, model from the paper:
"Rethinking the Inception Architecture for Computer Vision"
http://arxiv.org/abs/1512.00567
Original source:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py
License: http://www.apache.org/licenses/LICENSE-2.0

Download pretrained weights from:
https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/inception_v3.pkl
"""

import lasagne
import theano
import theano.tensor as t
from lasagne.layers import ConcatLayer, Conv2DLayer, DenseLayer, GlobalPoolLayer
from lasagne.layers import InputLayer, Pool2DLayer
from lasagne.layers.normalization import batch_norm
from lasagne.nonlinearities import softmax

new_pool_size = 3
new_pad = 1


def preprocess_input(x):
    # Expected input: RGB uint8 image
    # Input to network should be bc01, 299x299 pixels, scaled to [-1, 1].
    """

    Parameters
    ----------

    x: np.array

    Returns
    -------

    """

    return 2. * (x.astype('float32') / 255. - 0.5)


def bn_conv(input_layer, **kwargs):
    """

    Parameters
    ----------
    input_layer :
    kwargs :

    Returns
    -------

    """
    l = Conv2DLayer(input_layer, **kwargs)
    l = batch_norm(l, epsilon=0.001)
    return l


def inception_a(input_layer, nfilt):
    # Corresponds to a modified version of figure 5 in the paper
    """

    Parameters
    ----------
    input_layer :
    nfilt :

    Returns
    -------

    """
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=5, pad=2)

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=3, pad=1)

    l4 = Pool2DLayer(
        input_layer, pool_size=new_pool_size, stride=1, pad=new_pad,
        mode='average_exc_pad')
    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inception_b(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    """

    Parameters
    ----------
    input_layer :
    nfilt :

    Returns
    -------

    """
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=3, pad=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2, mode='average_exc_pad')

    return ConcatLayer([l1, l2, l3])


def inception_c(input_layer, nfilt):
    # Corresponds to figure 6 in the paper
    """

    Parameters
    ----------
    input_layer :
    nfilt :

    Returns
    -------

    """
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3))
    l3 = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3))

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inception_d(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    """

    Parameters
    ----------
    input_layer :
    nfilt :

    Returns
    -------

    """
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
    l1 = bn_conv(l1, num_filters=nfilt[0][1], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))
    l2 = bn_conv(l2, num_filters=nfilt[1][3], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2, mode='average_exc_pad')

    return ConcatLayer([l1, l2, l3])


def inception_e(input_layer, nfilt, pool_mode):
    # Corresponds to figure 7 in the paper
    """

    Parameters
    ----------
    input_layer :
    nfilt :
    pool_mode :

    Returns
    -------

    """
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2a = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1))
    l2b = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3a = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1))
    l3b = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0))

    l4 = Pool2DLayer(
        input_layer, pool_size=new_pool_size, stride=1, pad=new_pad, mode=pool_mode)

    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2a, l2b, l3a, l3b, l4])


def build_network():
    """

    Returns
    -------

    """

    input_var = t.tensor4('inputs')
    target = t.matrix('targets')
    lr = t.scalar('lr', dtype=theano.config.floatX)
    poolmode = 'average_exc_pad'
    new_pool_size2 = 3

    net = {'input': InputLayer((None, 3, 299, 299), input_var=input_var)}

    net['conv'] = bn_conv(net['input'],
                          num_filters=32, filter_size=3, stride=2)
    net['conv_1'] = bn_conv(net['conv'], num_filters=32, filter_size=3)
    net['conv_2'] = bn_conv(net['conv_1'],
                            num_filters=64, filter_size=3, pad=1)
    net['pool'] = Pool2DLayer(net['conv_2'], pool_size=new_pool_size2, stride=2,
                              mode=poolmode)

    net['conv_3'] = bn_conv(net['pool'], num_filters=80, filter_size=1)

    net['conv_4'] = bn_conv(net['conv_3'], num_filters=192, filter_size=3)

    net['pool_1'] = Pool2DLayer(net['conv_4'],
                                pool_size=new_pool_size2, stride=2, mode=poolmode)
    net['mixed/join'] = inception_a(
        net['pool_1'], nfilt=((64,), (48, 64), (64, 96, 96), (32,)))
    net['mixed_1/join'] = inception_a(
        net['mixed/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

    net['mixed_2/join'] = inception_a(
        net['mixed_1/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

    net['mixed_3/join'] = inception_b(
        net['mixed_2/join'], nfilt=((384,), (64, 96, 96)))

    net['mixed_4/join'] = inception_c(
        net['mixed_3/join'],
        nfilt=((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)))

    net['mixed_5/join'] = inception_c(
        net['mixed_4/join'],
        nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

    net['mixed_6/join'] = inception_c(
        net['mixed_5/join'],
        nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

    net['mixed_7/join'] = inception_c(
        net['mixed_6/join'],
        nfilt=((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)))

    net['mixed_8/join'] = inception_d(
        net['mixed_7/join'],
        nfilt=((192, 320), (192, 192, 192, 192)))

    net['mixed_9/join'] = inception_e(
        net['mixed_8/join'],
        nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
        pool_mode='average_exc_pad')

    net['mixed_10/join'] = inception_e(
        net['mixed_9/join'],
        nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
        pool_mode=poolmode)

    net['pool3'] = GlobalPoolLayer(net['mixed_10/join'])

    net['softmax'] = DenseLayer(
        net['pool3'], num_units=1008, nonlinearity=softmax)

    train_output = lasagne.layers.get_output(net['softmax'],
                                             deterministic=False)
    train_loss = lasagne.objectives.categorical_crossentropy(train_output,
                                                             target)
    train_loss = lasagne.objectives.aggregate(train_loss)
    train_err = t.mean(t.neq(t.argmax(train_output, axis=1),
                             t.argmax(target, axis=1)),
                       dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(net['softmax'], trainable=True)
    updates = lasagne.updates.sgd(loss_or_grads=train_loss, params=params,
                                  learning_rate=lr)

    test_output = lasagne.layers.get_output(net['softmax'], deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_output, target)
    test_loss = lasagne.objectives.aggregate(test_loss)
    test_err = t.mean(t.neq(t.argmax(test_output, axis=1),
                            t.argmax(target, axis=1)),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target, lr], [train_loss, train_err],
                               updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target], [test_loss, test_err])

    return {'model': net['softmax'], 'train_fn': train_fn, 'val_fn': val_fn}

if __name__ == "__main__":

    import json
    from keras.preprocessing.image import ImageDataGenerator
    from snntoolbox.datasets.utils import save_parameters

    data_path = '/home/rbodo/.snntoolbox/Datasets/imagenet'
    train_path = data_path + '/training'
    test_path = data_path + '/validation'
    save_path = data_path + '/GoogLeNet'
    class_idx_path = save_path + '/imagenet_class_index.json'

    # Training parameters
    target_size = (299, 299)
    batch_size = 20
    samples_per_epoch = 10000
    num_val_samples = 500
    num_epochs = 50

    # Decaying LR
    LR_start = 0.001  # 0.045
    LR_decay = 0.1  # 0.94

    print('Loading dataset...')

    class_idx = json.load(open(class_idx_path, "r"))
    classes = [class_idx[str(idx)][0] for idx in range(len(class_idx))]

    datagen = ImageDataGenerator()
    train_dataflow = datagen.flow_from_directory(train_path,
                                                 target_size=target_size,
                                                 classes=classes,
                                                 batch_size=batch_size)

    test_dataflow = datagen.flow_from_directory(test_path,
                                                target_size=target_size,
                                                classes=classes,
                                                batch_size=num_val_samples)

    print('Building the CNN...')
    model_dict = build_network()
    model = model_dict['model']
    train_func = model_dict['train_fn']
    val_func = model_dict['val_fn']

    # Initialize with pretrained weights.
    from snntoolbox.datasets.utils import load_parameters
    p = load_parameters('/home/rbodo/.snntoolbox/data/imagenet/'
                        'inception_averagepool/72.94_91.22.h5')
    lasagne.layers.set_all_param_values(model, p)

    print("Evaluating pre-trained model...")
    x_test, y_test = test_dataflow.next()
    x_test = preprocess_input(x_test)
    ls, err = val_func(x_test, y_test)
    print("Val. loss: {}, accuracy: {:.2%}".format(ls, 1 - err))
    del x_test, y_test

    print("Refining pre-trained model...")
    LR = LR_start
    for e in range(num_epochs):
        print("Epoch {}".format(e))
        batch_idx = ls = acc = 0
        for x_batch, y_batch in train_dataflow:
            x_batch = preprocess_input(x_batch)
            ls, err = train_func(x_batch, y_batch, LR)
            acc += 100 * (1 - err)
            batch_idx += 1
            print("Training loss: {}, accuracy: {}".format(ls, acc / batch_idx))
            if batch_idx * batch_size >= samples_per_epoch:
                break
        LR *= LR_decay
        print("Validating after epoch {}...".format(e))
        x_eval, y_eval = test_dataflow.next()
        x_eval = preprocess_input(x_eval)
        ls, err = val_func(x_eval, y_eval)
        acc = 100 * (1 - err)
        print("Val. loss: {}, accuracy: {}, lr: {}".format(ls, acc, LR))
        # Save network
        filepath = '/home/rbodo/.snntoolbox/data/imagenet/' \
                   'inception_averagepool/{}.h5'.format(acc)
        parameters = lasagne.layers.get_all_param_values(model)
        save_parameters(parameters, filepath)
