# coding=utf-8

"""py.test fixtures with module-scope."""

import numpy as np
import os
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, \
    Dropout, Dense, Concatenate, Activation, BatchNormalization
from tensorflow.keras.utils import to_categorical

from snntoolbox.bin.utils import update_setup
from snntoolbox.utils.utils import import_configparser, is_module_installed


@pytest.fixture(scope='function')
def _config(_path_wd, _datapath):
    return get_config(_path_wd, _datapath)


@pytest.fixture(scope='function')
def _config_first(_path_wd, _datapath_first):
    return get_config(_path_wd, _datapath_first)


@pytest.fixture(scope='function')
def _config_last(_path_wd, _datapath_last):
    return get_config(_path_wd, _datapath_last)


def get_config(path_wd, datapath):
    path_wd = str(path_wd)
    datapath = str(datapath)
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


def get_dataset(data_format=None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    if data_format is None:
        data_format = keras.backend.image_data_format()
    axis = 1 if data_format == 'channels_first' else -1
    x_train = np.expand_dims(x_train, axis)
    x_test = np.expand_dims(x_test, axis)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def save_dataset(datapath, x, y):
    np.savez_compressed(os.path.join(str(datapath), 'x_test'), x)
    np.savez_compressed(os.path.join(str(datapath), 'y_test'), y)
    np.savez_compressed(os.path.join(str(datapath), 'x_norm'), x)


def get_datapath(tmpdir_factory, data_format):
    datapath = tmpdir_factory.mktemp('dataset')
    _, _, x_test, y_test = get_dataset(data_format)
    save_dataset(datapath, x_test, y_test)
    return datapath


@pytest.fixture(scope='session')
def _datapath_first(tmpdir_factory):
    """Path to dataset stored in channel_first format."""
    return get_datapath(tmpdir_factory, 'channels_first')


@pytest.fixture(scope='session')
def _datapath_last(tmpdir_factory):
    """Path to dataset stored in channel_last format."""
    return get_datapath(tmpdir_factory, 'channels_last')


@pytest.fixture(scope='session')
def _datapath(_datapath_first, _datapath_last):
    if keras.backend.image_data_format() == 'channels_first':
        return _datapath_first
    return _datapath_last


@pytest.fixture(scope='session')
def _dataset(_dataset_first, _dataset_last):
    if keras.backend.image_data_format() == 'channels_first':
        return _dataset_first
    return _dataset_last


@pytest.fixture(scope='session')
def _dataset_first():
    return get_dataset('channels_first')


@pytest.fixture(scope='session')
def _dataset_last():
    return get_dataset('channels_last')


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
    return get_model_1(_dataset)


def get_model_1(dataset):
    x_train, y_train, x_test, y_test = dataset

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

    if not is_module_installed('keras_rewiring'):
        return

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


@pytest.fixture(scope='session')
def _model_4(_dataset):
    return get_model_4(_dataset)


@pytest.fixture(scope='session')
def _model_4_first(_dataset_first):
    return get_model_4(_dataset_first)


def get_model_4(dataset):

    if not is_module_installed('torch'):
        return

    import torch
    import torch.nn as nn
    from tests.parsing.models import pytorch

    x_train, y_train, x_test, y_test = dataset

    # Pytorch doesn't support one-hot labels.
    y_train = np.argmax(y_train, 1)
    y_test = np.argmax(y_test, 1)

    class PytorchDataset(torch.utils.data.Dataset):
        def __init__(self, data, target, transform=None):
            self.data = torch.from_numpy(data).float()
            self.target = torch.from_numpy(target).long()
            self.transform = transform

        def __getitem__(self, index):
            x = self.data[index]

            if self.transform:
                x = self.transform(x)

            return x, self.target[index]

        def __len__(self):
            return len(self.data)

    trainset = torch.utils.data.DataLoader(PytorchDataset(x_train, y_train),
                                           batch_size=64)
    testset = torch.utils.data.DataLoader(PytorchDataset(x_test, y_test),
                                          batch_size=64)

    model = pytorch.Model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    acc = 0
    for epoch in range(3):
        for i, (xx, y) in enumerate(trainset):
            optimizer.zero_grad()
            outputs = model(xx)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        total = 0
        correct = 0
        with torch.no_grad():
            for xx, y in testset:
                outputs = model(xx)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        acc = correct / total

    print("Test accuracy: {:.2%}".format(acc))
    # assert acc > 0.96, "Test accuracy after training not high enough."

    return model


spinnaker_conditions = (is_module_installed('keras_rewiring') and
                        is_module_installed('pynn_object_serialisation') and
                        (is_module_installed('pyNN.spiNNaker') or
                         is_module_installed('spynnaker8')))
spinnaker_skip_if_dependency_missing = pytest.mark.skipif(
    not spinnaker_conditions, reason="Spinnaker dependency missing.")

nest_conditions = (is_module_installed('pyNN') and
                   is_module_installed('nest'))
nest_skip_if_dependency_missing = pytest.mark.skipif(
    not nest_conditions, reason="Nest dependency missing.")

brian2_conditions = (is_module_installed('brian2'))
brian2_skip_if_dependency_missing = pytest.mark.skipif(
    not brian2_conditions, reason="Brian2 dependency missing.")

pytorch_conditions = (is_module_installed('torch') and
                      is_module_installed('onnx') and
                      is_module_installed('onnx2keras') and
                      len(tf.config.list_physical_devices('GPU')))
pytorch_skip_if_dependency_missing = pytest.mark.skipif(
    not pytorch_conditions, reason='Pytorch dependencies missing.')

loihi_conditions = (is_module_installed('nxsdk') and
                    is_module_installed('nxsdk_modules_ncl'))
loihi_skip_if_dependency_missing = pytest.mark.skipif(
    not loihi_conditions, reason='Loihi dependency missing.')


def get_examples():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                        'examples'))
    files = ['mnist_keras_INI.py']
    if brian2_conditions:
        files.append('mnist_keras_brian2.py')
    if nest_conditions:
        files.append('mnist_keras_nest.py')
    if spinnaker_conditions:
        files.append('mnist_keras_spiNNaker.py')
        files.append('mnist_keras_spiNNaker_sparse.py')
    if loihi_conditions:
        files.append('mnist_keras_loihi.py')

    example_filepaths = [os.path.join(path, f) for f in files]

    return example_filepaths


@pytest.fixture(scope='session', params=get_examples())
def _example_filepath(request):
    return request.param
