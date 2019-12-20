# coding=utf-8

"""py.test fixtures with module-scope."""

import os

import keras
import numpy as np
import pytest
from keras import Input, Model
from keras.datasets import mnist
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dropout, Dense, \
    Concatenate, Activation, BatchNormalization
from keras.utils import np_utils

from snntoolbox.bin.utils import update_setup
from snntoolbox.utils.utils import import_configparser


def is_module_installed(mod):
    import sys
    if sys.version_info[0] < 3:
        import pkgutil
        return pkgutil.find_loader(mod) is not None
    else:
        import importlib
        return importlib.util.find_spec(mod) is not None


@pytest.fixture(scope='function')
def _config(_path_wd, _datapath):

    path_wd = str(_path_wd)
    datapath = str(_datapath)
    filename_ann = 'mnist_cnn'
    configparser = import_configparser()
    config = configparser.ConfigParser()

    config.read_dict({'paths': {'path_wd': path_wd,
                                'dataset_path': datapath,
                                'filename_ann': filename_ann}})

    with open(os.path.join(path_wd, filename_ann + '.h5'), 'w'):
        pass

    config_filepath = os.path.join(path_wd, 'config')
    with open(config_filepath, 'w') as configfile:
        config.write(configfile)

    config = update_setup(config_filepath)

    return config


@pytest.fixture(scope='function')
def _path_wd(tmpdir_factory):
    return tmpdir_factory.mktemp('wd')


@pytest.fixture(scope='session')
def _datapath(tmpdir_factory, _dataset):
    datapath = tmpdir_factory.mktemp('dataset')
    x_train, y_train, x_test, y_test = _dataset

    np.savez_compressed(os.path.join(datapath, 'x_test'), x_test)
    np.savez_compressed(os.path.join(datapath, 'y_test'), y_test)
    np.savez_compressed(os.path.join(datapath, 'x_norm'), x_test)

    return datapath


@pytest.fixture(scope='session')
def _dataset():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    x_train = np.expand_dims(x_train, axis)
    x_test = np.expand_dims(x_test, axis)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


@pytest.fixture(scope='session')
def _normset(_dataset):
    return _dataset[2]


@pytest.fixture(scope='session')
def _testset(_dataset):
    return _dataset[2:]


@pytest.fixture(scope='session')
def _installed_input_libs(_config):
    input_libs = _config.get('restrictions', 'model_libs')
    return [lib for lib in input_libs if is_module_installed(lib)]


@pytest.fixture(scope='session')
def _model_1(_dataset):

    x_train, y_train, x_test, y_test = _dataset

    input_shape = x_train.shape[1:]
    input_layer = Input(input_shape)

    layer = Conv2D(filters=16,
                   kernel_size=(5, 5),
                   strides=(2, 2),
                   activation='relu',
                   use_bias=False)(input_layer)
    layer = Conv2D(filters=32,
                   kernel_size=(3, 3),
                   activation='relu',
                   use_bias=False)(layer)
    layer = AveragePooling2D()(layer)
    layer = Conv2D(filters=8,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   use_bias=False)(layer)
    layer = Flatten()(layer)
    layer = Dropout(0.01)(layer)
    layer = Dense(units=10,
                  activation='softmax',
                  use_bias=False)(layer)

    model = Model(input_layer, layer)

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    history = model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2,
                        validation_data=(x_test, y_test))

    assert history.history['val_accuracy'][-1] > 0.95

    return model


@pytest.fixture(scope='session')
def _model_2(_dataset):
    x_train, y_train, x_test, y_test = _dataset

    axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

    input_shape = x_train.shape[1:]
    input_layer = Input(input_shape)

    layer = Conv2D(filters=16,
                   kernel_size=(5, 5),
                   strides=(2, 2))(input_layer)
    layer = BatchNormalization(axis=axis)(layer)
    layer = Activation('relu')(layer)
    layer = AveragePooling2D()(layer)
    branch1 = Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same',
                     activation='relu')(layer)
    branch2 = Conv2D(filters=8,
                     kernel_size=(1, 1),
                     activation='relu')(layer)
    layer = Concatenate(axis=axis)([branch1, branch2])
    layer = Conv2D(filters=10,
                   kernel_size=(3, 3),
                   activation='relu')(layer)
    layer = Flatten()(layer)
    layer = Dropout(1e-5)(layer)
    layer = Dense(units=10,
                  activation='softmax')(layer)

    model = Model(input_layer, layer)

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    # Train model with backprop.
    history = model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2,
                        validation_data=(x_test, y_test))

    assert history.history['val_accuracy'][-1] > 0.96

    return model


@pytest.fixture(scope='session')
def _model_3(_dataset):
    from keras_rewiring import Sparse, SparseConv2D, SparseDepthwiseConv2D

    x_train, y_train, x_test, y_test = _dataset

    axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

    input_shape = x_train.shape[1:]
    input_layer = Input(input_shape)

    layer = SparseConv2D(filters=16,
                         kernel_size=(5, 5),
                         strides=(2, 2))(input_layer)
    layer = BatchNormalization(axis=axis)(layer)
    layer = Activation('relu')(layer)
    layer = AveragePooling2D()(layer)
    branch1 = SparseConv2D(filters=32,
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(layer)
    branch2 = SparseDepthwiseConv2D(kernel_size=(1, 1),
                                    activation='relu')(layer)
    layer = Concatenate(axis=axis)([branch1, branch2])
    layer = SparseConv2D(filters=10,
                         kernel_size=(3, 3),
                         activation='relu')(layer)
    layer = Flatten()(layer)
    layer = Dropout(1e-5)(layer)
    layer = Sparse(units=10,
                   activation='softmax')(layer)

    model = Model(input_layer, layer)

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    # Train model with backprop.
    history = model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2,
                        validation_data=(x_test, y_test))

    assert history.history['val_accuracy'][-1] > 0.96

    return model


def get_examples():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                        'examples'))
    files = ['mnist_keras_INI.py']
    if is_module_installed('brian2'):
        files.append('mnist_keras_brian2.py')
    if is_module_installed('pyNN') and is_module_installed('nest'):
        files.append('mnist_keras_nest.py')

    example_filepaths = [os.path.join(path, f) for f in files]

    return example_filepaths


@pytest.fixture(scope='session', params=get_examples())
def _example_filepath(request):
    return request.param
