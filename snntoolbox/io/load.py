# -*- coding: utf-8 -*-
"""
Functions to load various properties of interest in analog and spiking neural
networks from disk.

Created on Wed Nov 18 13:38:46 2015

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library
from builtins import open

import os
import sys
import numpy as np
from six.moves import cPickle
from snntoolbox import echo
from snntoolbox.config import globalparams, architectures, datasets

standard_library.install_aliases()


class ANN():
    """
    Represent a neural network.

    Implements a class that contains essential information about the
    architecture and weights of a neural network, while being independent of
    the library from which the original network was built.

    The constructor ``__init__()`` performs the extraction of the
    attributes of interest.

    Parameters
    ----------

        model : network object
            A network object of the ``model_lib`` language, e.g. keras.

    Attributes
    ----------

        - weights : array
            Weights connecting the input layer.

        - biases : array
            Biases of the network. For conversion to spiking nets, zero biases
            are found to work best.

        - input_shape : list
            The dimensions of the input sample.

        - layers : list
            List of all the layers of the network, where each layer contains a
            dictionary with keys

            - layer_num : int
                Index of layer.

            - layer_type : string
                Describing the type, e.g. `Dense`, `Convolution`, `Pool`.

            - output_shape : list
                The output dimensions of the layer.

            In addition, `Dense` and `Convolution` layer types contain

            - weights : array
                The weight parameters connecting the layer with the next.

            `Convolution` layers contain further

            - nb_col : int
                The x-dimension of filters.

            - nb_row : int
                The y-dimension of filters.

            - border_mode : string
                How to handle borders during convolution, e.g. `full`, `valid`,
                `same`.

            `Pooling` layers contain

            - pool_size : list
                Specifies the subsampling factor in each dimension.

            - strides : list
                The stepsize in each dimension during pooling.

    The initializer is meant to be extended by functionality to extract any
    model written in any of the common neural network libraries, e.g. keras,
    theano, caffe, torch, etc.

    The simplified returned network can then be used in the SNN conversion and
    simulation toolbox.

    """

    def __init__(self, model):
        bn_layers = {'Dense', 'Convolution2D'}
        if globalparams['model_lib'] == 'keras':
            layers = model.layers
            self.input_shape = model.input_shape
            self.layers = []

            # Label the input layer according to its shape.
            if globalparams['architecture'] == 'mlp':
                self.labels = ['InputLayer_{}'.format(self.input_shape[1])]
            else:
                self.labels = ['InputLayer_{}x{}x{}'.format(
                                self.input_shape[1],
                                self.input_shape[2],
                                self.input_shape[3])]
            for (layer_num, layer) in enumerate(layers):
                attributes = {'layer_num': layer_num,
                              'layer_type': layer.__class__.__name__,
                              'output_shape': layer.output_shape}
                if len(attributes['output_shape']) == 2:
                    shape_string = '_{}'.format(attributes['output_shape'][1])
                else:
                    shape_string = '_{}x{}x{}'.format(
                        attributes['output_shape'][1],
                        attributes['output_shape'][2],
                        attributes['output_shape'][3])
                num_str = str(layer_num) if layer_num > 9 else \
                    '0' + str(layer_num)
                self.labels.append(num_str + attributes['layer_type'] +
                                   shape_string)
                attributes.update({'label': self.labels[-1]})

                next_layer = layers[layer_num + 1] \
                    if layer_num + 1 < len(layers) else None
                next_layer_name = next_layer.__class__.__name__ \
                    if next_layer else None
                if next_layer_name == 'BatchNormalization' and \
                        attributes['layer_type'] not in bn_layers:
                    raise NotImplementedError(
                        "A batchnormalization layer must follow a layer of " +
                        "type {}, not {}.".format(bn_layers,
                                                  attributes['layer_type']))
                if attributes['layer_type'] in {'Dense', 'Convolution2D'}:
                    wb = layer.get_weights()
                    if next_layer_name == 'BatchNormalization':
                        weights = next_layer.get_weights()
                        wb = absorb_bn(wb[0], wb[1],  # W, b
                                       weights[0],  # gamma
                                       weights[1],  # beta
                                       weights[2],  # mean
                                       weights[3],  # std
                                       next_layer.epsilon)
                    attributes.update({'weights': wb})
                if attributes['layer_type'] == 'Convolution2D':
                    attributes.update({'input_shape': layer.input_shape,
                                       'nb_filter': layer.nb_filter,
                                       'nb_col': layer.nb_col,
                                       'nb_row': layer.nb_row,
                                       'border_mode': layer.border_mode})
                elif attributes['layer_type'] in {'MaxPooling2D',
                                                  'AveragePooling2D'}:
                    attributes.update({'input_shape': layer.input_shape,
                                       'pool_size': layer.pool_size,
                                       'strides': layer.strides,
                                       'border_mode': layer.border_mode})
                self.layers.append(attributes)
        elif globalparams['model_lib'] == 'lasagne':
            import lasagne
            layers = lasagne.layers.get_all_layers(model)
            weights = lasagne.layers.get_all_param_values(model)
            self.input_shape = layers[0].shape
            self.layers = []

            # Label the input layer according to its shape.
            if globalparams['architecture'] == 'mlp':
                self.labels = ['InputLayer_{}'.format(self.input_shape[1])]
            else:
                self.labels = ['InputLayer_{}x{}x{}'.format(
                                self.input_shape[1],
                                self.input_shape[2],
                                self.input_shape[3])]
            weights_idx = 0
            for (layer_num, layer) in enumerate(layers):
                name = layer.__class__.__name__
                if name == 'DenseLayer':
                    layer_type = 'Dense'
                elif name in {'Conv2DLayer', 'Conv2DDNNLayer'}:
                    layer_type = 'Convolution2D'
                elif name == 'MaxPool2DLayer':
                    layer_type = 'MaxPooling2D'
                elif name in {'Pool2DLayer'}:
                    layer_type = 'AveragePooling2D'
                elif name == 'DropoutLayer':
                    layer_type = 'Dropout'
                elif name == 'FlattenLayer':
                    layer_type = 'Flatten'
                elif name == 'BatchNormLayer':
                    layer_type = 'BatchNorm'
                elif name == 'InputLayer':
                    continue
                else:
                    layer_type = layer.__class__.__name__
                attributes = {'layer_num': layer_num,
                              'layer_type': layer_type,
                              'output_shape': layer.output_shape}
                next_layer = layers[layer_num + 1] \
                    if layer_num + 1 < len(layers) else None
                next_layer_name = next_layer.__class__.__name__ \
                    if next_layer else None
                if next_layer_name == 'BatchNormLayer' and \
                        layer_type not in bn_layers:
                    raise NotImplementedError("A batchnormalization layer " +
                                              "must follow a layer of type " +
                                              "{}, not {}.".format(bn_layers,
                                                                   layer_type))
                if attributes['layer_type'] in {'Dense', 'Convolution2D'}:
                    wb = weights[weights_idx: weights_idx + 2]
                    weights_idx += 2  # For weights and biases
                    if next_layer_name == 'BatchNormLayer':
                        wb = absorb_bn(wb[0], wb[1],  # W, b
                                       weights[weights_idx + 0],  # gamma
                                       weights[weights_idx + 1],  # beta
                                       weights[weights_idx + 2],  # mean
                                       weights[weights_idx + 3],  # std
                                       next_layer.epsilon)
                        weights_idx += 4
                    attributes.update({'weights': wb})
                if attributes['layer_type'] == 'Convolution2D':
                    fs = layer.filter_size
                    if layer.pad == (0, 0):
                        border_mode = 'valid'
                    elif layer.pad == (fs[0] // 2, fs[1] // 2):
                        border_mode = 'same'
                    elif layer.pad == (fs[0] - 1, fs[1] - 1):
                        border_mode = 'full'
                    else:
                        raise NotImplementedError("Padding {} ".format(
                            layer.pad) + "could not be interpreted as any " +
                            "of the supported border modes 'valid', 'same' " +
                            "or 'full'.")
                    attributes.update({'input_shape': layer.input_shape,
                                       'nb_filter': layer.num_filters,
                                       'nb_col': fs[1],
                                       'nb_row': fs[0],
                                       'border_mode': border_mode})
                elif attributes['layer_type'] in {'MaxPooling2D',
                                                  'AveragePooling2D'}:
                    ps = layer.pool_size
                    if layer.pad == (0, 0):
                        border_mode = 'valid'
                    elif layer.pad == (ps[0] // 2, ps[1] // 2):
                        border_mode = 'same'
                    elif layer.pad == (ps[0] - 1, ps[1] - 1):
                        border_mode = 'full'
                    else:
                        raise NotImplementedError("Padding {} ".format(
                            layer.pad) + "could not be interpreted as any " +
                            "of the supported border modes 'valid', 'same' " +
                            "or 'full'.")
                    attributes.update({'input_shape': layer.input_shape,
                                       'pool_size': layer.pool_size,
                                       'strides': layer.stride,
                                       'border_mode': border_mode})
                # Append layer label
                if len(attributes['output_shape']) == 2:
                    shape_string = '_{}'.format(attributes['output_shape'][1])
                else:
                    shape_string = '_{}x{}x{}'.format(
                        attributes['output_shape'][1],
                        attributes['output_shape'][2],
                        attributes['output_shape'][3])
                num_str = str(layer_num) if layer_num > 9 else \
                    '0' + str(layer_num)
                self.labels.append(num_str + attributes['layer_type'] +
                                   shape_string)
                attributes.update({'label': self.labels[-1]})
                # Append layer
                self.layers.append(attributes)

    def get_config(self):
        layer_config = []
        for layer in self.layers:
            layer_config.append([{key: layer[key]} for key in layer.keys()
                                 if key not in ['weights']])
        config = {'name': self.__class__.__name__,
                  'input_shape': self.input_shape,
                  'layers': layer_config}
        return config


def absorb_bn(w, b, gamma, beta, mean, std, epsilon):
    ax = 0 if w.ndim < 3 else 1
    w_normed = w*np.swapaxes(gamma, 0, ax) / (np.swapaxes(std, 0, ax)+epsilon)
    b_normed = ((b - mean.flatten()) * gamma.flatten() /
                (std.flatten() + epsilon) + beta.flatten())
    return [w_normed, b_normed]


def load_model(filename=None, spiking=False):
    """
    Load model architecture.

    Parameters
    ----------

    filename : string, optional
        If no filename is given, the method assumes the model is named
        ``globalparams['filename']``. This will be true most of the time; it
        can be useful to specify a different name e.g. when loading a
        normalized net, where the filename is appended with ``_normWeights``.
    spiking : boolean, optional
        Tells the function if the model to load is a spiking network. In that
        case, various initialization steps are performed, depending on the
        simulator.

    Returns
    -------

    model : dict
        If ``spiking==False``, the returned dictionary contains a key 'model'
        with value the network model object in the ``model_lib`` language, e.g.
        keras. If globalparams['model_lib']=='lasagne', the ``model`` dict
        additionally contains a train and test function (keys 'train_fn',
        'val_fn').
        In case ``spiking==True``, a converted spiking network is loaded from
        disk, and the ``model`` dict contains a key 'layers' with pyNN layers
        (if a pyNN simulator is used), or a keras 'model' and theano
        'get_output' function, if the builtin INI simulator is used.
    """

    if filename is None:
        filename = globalparams['filename']
    if spiking:
        return load_snn(filename)
    else:
        return load_ann(filename)


def load_snn(filename):
    import snntoolbox
    from snntoolbox import sim

    if snntoolbox._SIMULATOR in snntoolbox.simulators_pyNN:
        layers = load_assembly()
        for i in range(len(layers)-1):
            filename = os.path.join(globalparams['path'],
                                    layers[i].label + '_' +
                                    layers[i+1].label)
            if os.path.isfile(filename):
                sim.Projection(layers[i], layers[i+1],
                               sim.FromFileConnector(filename))
            else:
                echo("Connections were not found at specified location.\n")
        return {'layers': layers}
    elif snntoolbox._SIMULATOR == 'INI':
        import theano
        import theano.tensor as T
        from keras import models
        from snntoolbox.config import cellparams

        path = os.path.join(globalparams['path'], filename + '.json')
        model = models.model_from_json(open(path).read(),
                                       custom_objects=snntoolbox.custom_layers)
        model.load_weights(os.path.join(globalparams['path'], filename+'.h5'))
        # Allocate input variables
        input_time = T.scalar('time')
        input_shape = list(model.input_shape)
        input_shape[0] = globalparams['batch_size']
        model.layers[0].batch_input_shape = input_shape
        kwargs = {'time_var': input_time}
        for layer in model.layers:
            sim.init_neurons(layer,
                             v_thresh=cellparams['v_thresh'],
                             tau_refrac=cellparams['tau_refrac'],
                             **kwargs)
            kwargs = {}
        # Compile model
        # Todo: Allow user to specify loss function here (optimizer is not
        # relevant as we do not train any more). Unfortunately, Keras does not
        # save these parameters. They can be obtained from the compiled model
        # by calling 'model.loss' and 'model.optimizer'.
        model.compile(loss='categorical_crossentropy', optimizer='sgd',
                      metrics=['accuracy'])
        output_spikes = model.layers[-1].get_output()
        output_time = sim.get_time(model.layers[-1])
        updates = sim.get_updates(model.layers[-1])
        get_output = theano.function([model.input, input_time],
                                     [output_spikes, output_time],
                                     updates=updates)
        return {'model': model, 'get_output': get_output}


def load_ann(filename):
    import snntoolbox

    if globalparams['model_lib'] == 'keras':
        if globalparams['dataset'] == 'caltech101':
            model = model_from_py(filename)['model']
        else:
            from keras import models
            path = os.path.join(globalparams['path'], filename + '.json')
            model = models.model_from_json(
                open(path).read(), custom_objects=snntoolbox.custom_layers)
        model.load_weights(os.path.join(globalparams['path'],
                                        filename + '.h5'))
        # Todo: Allow user to specify loss function here (optimizer is not
        # relevant as we do not train any more). Unfortunately, Keras does not
        # save these parameters. They can be obtained from the compiled model
        # by calling 'model.loss' and 'model.optimizer'.
        model.compile(loss='categorical_crossentropy', optimizer='sgd',
                      metrics=['accuracy'])
        return {'model': model}
    elif globalparams['model_lib'] == 'lasagne':
        return model_from_py(filename)


def model_from_py(filename):
    # Import script which builds the model
    v = sys.version_info
    if v >= (3, 5):
        import importlib.util
        path = os.path.join(globalparams['path'], filename + '.py')
        spec = importlib.util.spec_from_file_location(filename, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    elif v >= (3, 3):
        from importlib.machinery import SourceFileLoader
        mod = SourceFileLoader(filename, path).load_module()
    else:
        import imp
        mod = imp.load_source(filename, path)

    if globalparams['model_lib'] == 'keras':
        return {'model': mod.build_network()}
    elif globalparams['model_lib'] == 'lasagne':
        import lasagne
        model, train_fn, val_fn = mod.build_network()
        params = load_weights(os.path.join(globalparams['path'],
                                           filename + '.h5'))
        lasagne.layers.set_all_param_values(model, params)
        return {'model': model, 'train_fn': train_fn, 'val_fn': val_fn}


def load_weights(filepath):
    """
    Load all layer weights from a HDF5 file.
    """

    import h5py

    f = h5py.File(filepath, mode='r')

    params = []
    for k in f.keys():
        params.append(np.array(f.get(k)))

    f.close()

    return params


def to_categorical(y, nb_classes):
    """
    Convert class vector to binary class matrix.

    If the input ``y`` has shape (``nb_samples``,) and contains integers from 0
    to ``nb_classes``, the output array will be of dimension
    (``nb_samples``, ``nb_classes``).
    """

    y = np.asarray(y, dtype='int32')
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def get_dataset():
    """
    Load a classification dataset.

    Returns
    -------
    dataset : tuple
        The dataset as a tuple containing the training and test sample arrays
        (X_train, Y_train, X_test, Y_test)

    Todo
    ----

    @Iulia: Discuss how to support non-classification datasets.
    """

    import gzip

    assert globalparams['dataset'] in datasets, \
        "Dataset {} not known. Supported datasets: {}".format(
                                            globalparams['dataset'], datasets)

    nb_classes = 10

    if globalparams['dataset'] == 'mnist':
        fname = globalparams['dataset'] + '.pkl.gz'
        path = download_dataset(
                fname, origin='https://s3.amazonaws.com/img-datasets/' + fname)

        if path.endswith('.gz'):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')

        if sys.version_info < (3,):
            (X_train, y_train), (X_test, y_test) = cPickle.load(f)
        else:
            (X_train, y_train), (X_test, y_test) = \
                cPickle.load(f, encoding='bytes')

        f.close()

    elif globalparams['dataset'] == 'cifar10':
        from keras.datasets import cifar10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    elif globalparams['dataset'] == 'caltech101':
        from snntoolbox.io.datasets import caltech101
        nb_classes = 102

        # Download & untar or get local path
        base_path = caltech101.download(dataset='img-gen-resized')

        # Path to image folder
        base_path = os.path.join(base_path, caltech101.tar_inner_dirname)

        # X_test contains only paths to images
        (X_test, y_test) = caltech101.load_paths_from_files(base_path,
                                                            'X_test.txt',
                                                            'y_test.txt')
        (X_train, y_train), (X_val, y_val) = caltech101.load_cv_split_paths(
                                                                base_path, 0)
        X_train = caltech101.load_samples(X_train, 2*globalparams['batch_size'])
        X_test = caltech101.load_samples(X_test, 2*globalparams['batch_size'])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)

    return (X_train, Y_train, X_test, Y_test)


def get_reshaped_dataset():
    """
    Load a classification dataset and shape it to fit a specific network model.

    Returns
    -------

    The dataset as a tuple containing the training and test sample arrays
    (X_train, Y_train, X_test, Y_test).
    With data of the form (channels, num_rows, num_cols), ``X_train`` and
    ``X_test`` have dimension (num_samples, channels*num_rows*num_cols)
    for a multi-layer perceptron, and
    (num_samples, channels, num_rows, num_cols) for a convolutional net.
    ``Y_train`` and ``Y_test`` have dimension (num_samples, num_classes).
    """

    (X_train, Y_train, X_test, Y_test) = get_dataset()

    assert globalparams['architecture'] in architectures, "Network \
        architecture {} not understood. Supported architectures: {}".format(
            globalparams['architecture'], architectures)
    if globalparams['architecture'] == 'mlp':
        X_train = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
        X_test = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))
    # Data container has no channel dimension, but we need 4D input for CNN:
    elif globalparams['architecture'] == 'cnn' and X_train.ndim < 4:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1],
                                  X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1],
                                X_test.shape[2])
    return (X_train, Y_train, X_test, Y_test)


def load_assembly():
    """
    Loads the populations in an assembly that was saved with the
    ``snntoolbox.io.save.save_assembly`` function.

    The term "assembly" refers
    to pyNN internal nomenclature, where ``Assembly`` is a collection of
    layers (``Populations``), which in turn consist of a number of neurons
    (``cells``).

    Returns
    -------

    populations : list
        List of pyNN ``Population`` objects
    """

    from snntoolbox import sim

    file = os.path.join(globalparams['path'],
                        'snn_' + globalparams['filename'])
    if os.path.isfile(file):
        if sys.version_info < (3,):
            s = cPickle.load(open(file, 'rb'))
        else:
            s = cPickle.load(open(file, 'rb'), encoding='bytes')
        # Iterate over populations in assembly
        populations = []
        for label in s['labels']:
            celltype = getattr(sim, s[label]['celltype'])
            population = getattr(sim, 'Population')(
                s[label]['size'], celltype, celltype.default_parameters,
                structure=s[label]['structure'], label=label)
            # Set the rest of the specified variables, if any.
            for variable in s['variables']:
                if getattr(population, variable, None) is None:
                    setattr(population, variable, s[label][variable])
            populations.append(population)
        if globalparams['verbose'] > 1:
            echo("Loaded spiking neuron layers from {}.\n".format(file))
        return populations
    else:
        echo("Spiking neuron layers were not found at specified location.\n")
        return []


def load_activations():
    """
    Read network activations from file.

    The method attempts to import the file
    ``<path>/<dataset>/<architecture>/<filename>/activations.json``
    containing the activations of the ANN.
    """

    file = os.path.join(globalparams['path'], 'activations')
    if os.path.isfile(file):
        if sys.version_info < (3,):
            activations = cPickle.load(open(file, 'rb'))
        else:
            activations = cPickle.load(open(file, 'rb'), encoding='bytes')
    else:
        echo("Activations were not found. Call the 'normalize_weights'" +
             "function on the ANN to create an activation file.\n")
        return []

    return activations


def download_dataset(fname, origin, untar=False):
    """
    Download a dataset, if not already there.

    Parameters
    ----------

    fname : string
        Full filename of dataset, e.g. ``mnist.pkl.gz``.
    origin : string
        Location of dataset, e.g. url
        https://s3.amazonaws.com/img-datasets/mnist.pkl.gz
    untar : boolean, optional
        If true, untar file.

    Returns
    -------

    fpath : string
        The path to the downloaded dataset. If the user has write access to
        ``home``, the dataset will be stored in ``~/.snntoolbox/datasets/``,
        otherwise in ``/tmp/.snntoolbox/datasets/``

    Todo
    ----

    Test under python2.
    """

    import tarfile
    import shutil
    from six.moves.urllib.error import URLError, HTTPError
    # Under Python 2, 'urlretrieve' relies on FancyURLopener from legacy
    # urllib module, known to have issues with proxy management
    from six.moves.urllib.request import urlretrieve

    datadir_base = os.path.expanduser(os.path.join('~', '.snntoolbox'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.snntoolbox')
    datadir = os.path.join(datadir_base, 'datasets')
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        echo("Downloading data from {}\n".format(origin))
        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise e

    if untar:
        if not os.path.exists(untar_fpath):
            echo("Untaring file...\n")
            tfile = tarfile.open(fpath, 'r:gz')
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise e
            tfile.close()
        return untar_fpath

    return fpath
