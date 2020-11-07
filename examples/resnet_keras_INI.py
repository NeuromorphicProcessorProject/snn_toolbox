"""End-to-end example for SNN Toolbox.

This script sets up a small ResNet using Keras, trains it for one epoch on
MNIST, stores model and dataset in a temporary folder on disk, creates a
configuration file for SNN toolbox, and finally calls the main function of SNN
toolbox to convert the trained ANN to an SNN and run it using INI simulator.
"""

import os
import time
import numpy as np

from tensorflow.keras import Input, Model, layers, backend, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser

# WORKING DIRECTORY #
#####################

# Define path where model and output files will be stored.
# The user is responsible for cleaning up this temporary directory.
path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp', str(time.time())))
os.makedirs(path_wd)

# GET DATASET #
###############

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input so we can train ANN with it.
# Will be converted back to integers for SNN layer.
x_train = x_train / 255
x_test = x_test / 255

# Add a channel dimension.
axis = 1 if backend.image_data_format() == 'channels_first' else -1
x_train = np.expand_dims(x_train, axis)
x_test = np.expand_dims(x_test, axis)

# One-hot encode target vectors.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Save dataset so SNN toolbox can find it.
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
# SNN toolbox will not do any training, but we save a subset of the training
# set so the toolbox can use it when normalizing the network parameters.
np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train[::10])


# CREATE ANN #
##############

# This section creates a small ResNet using Keras, and trains it with
# backpropagation. There are no spikes involved at this point.

# Here we copy a residual block as implemented in
# ``tensorflow.keras.applications.resnet`` but make one change: The final conv
# layer needs a ReLU activation as explained below.
def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.

    Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

    Returns:
    Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    # Swapped order of Activation and Add layers so that ReLU is applied to
    # the previous conv layer. The sequence {Conv + ReLU} or {Conv + BN + ReLU}
    # is necessary because the SNN spike generation implicitly applies ReLU on
    # each spiking layer.
    x = layers.Activation('relu', name=name + '_out')(x)
    x = layers.Add(name=name + '_add')([shortcut, x])
    return x


input_shape = x_train.shape[1:]
input_layer = Input(input_shape)
layer = layers.Conv2D(32, 3, strides=2, activation='relu')(input_layer)
layer = block1(layer, filters=8, conv_shortcut=False, name='block1')
layer = layers.Flatten()(layer)
layer = layers.Dense(units=10, activation='softmax')(layer)

model = Model(input_layer, layer)

model.summary()

model.compile('adam', 'categorical_crossentropy', ['accuracy'])

# Train model with backprop.
model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2,
          validation_data=(x_test, y_test))

# Store model so SNN Toolbox can find it.
model_name = 'mnist_resnet'
models.save_model(model, os.path.join(path_wd, model_name + '.h5'))

# SNN TOOLBOX CONFIGURATION #
#############################

# Create a config file with experimental setup for SNN Toolbox.
configparser = import_configparser()
config = configparser.ConfigParser()

config['paths'] = {
    'path_wd': path_wd,  # Path to model.
    'dataset_path': path_wd,  # Path to dataset.
    'filename_ann': model_name  # Name of input model.
}

config['tools'] = {
    'evaluate_ann': True,  # Test ANN on dataset before conversion.
    'normalize': True  # Normalize weights for full dynamic range.
}

config['simulation'] = {
    'simulator': 'INI',  # Chooses execution backend of SNN toolbox.
    'duration': 50,  # Number of time steps to run each sample.
    'num_to_test': 100,  # How many test samples to run.
    'batch_size': 50,  # Batch size for simulation.
    'keras_backend': 'tensorflow'  # Which keras backend to use.
}

config['output'] = {
    'plot_vars': {  # Various plots (slows down simulation).
        'spiketrains',  # Leave section empty to turn off plots.
        'spikerates',
        'activations',
        'correlation',
        'v_mem',
        'error_t'}
}

# Store config file.
config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# RUN SNN TOOLBOX #
###################

main(config_filepath)
