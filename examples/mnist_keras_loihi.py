"""End-to-end example for SNN Toolbox.

This script sets up a small CNN using Keras and tensorflow, trains it for one
epoch on MNIST, stores model and dataset in a temporary folder on disk, creates
a configuration file for SNN toolbox, and finally calls the main function of
SNN toolbox to convert the trained ANN to an SNN and run it using Intel's Loihi
processor.
"""

import os
import sys
import time
import numpy as np

from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, \
    Dropout, BatchNormalization, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser

# Enable SLURM to run network on Loihi.
os.environ['SLURM'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

# Add path to the nxsdk_modules package.
sys.path.append('/homes/rbodo/Repositories/nxsdk_modules')

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
axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
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

# This section creates a CNN using Keras, and trains it with backpropagation.
# There are no spikes involved at this point. The model is far more complicated
# than necessary for MNIST, but serves the purpose to illustrate the kind of
# layers and topologies supported (in particular branches).

input_shape = x_train.shape[1:]
input_layer = Input(input_shape)

layer = Conv2D(filters=16,
               kernel_size=(5, 5),
               strides=(2, 2))(input_layer)
layer = BatchNormalization(axis=axis)(layer)
layer = Activation('relu')(layer)
layer = Conv2D(filters=32,
               kernel_size=(3, 3),
               activation='relu')(layer)
layer = AveragePooling2D()(layer)
layer = Conv2D(filters=8,
               kernel_size=(3, 3),
               padding='same',
               activation='relu')(layer)
layer = Flatten()(layer)
layer = Dropout(0.01)(layer)
layer = Dense(units=10,
              activation='softmax')(layer)

model = Model(input_layer, layer)

model.summary()

model.compile('adam', 'categorical_crossentropy', ['accuracy'])

# Train model with backprop.
model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2,
          validation_data=(x_test, y_test))

# Store model so SNN Toolbox can find it.
model_name = 'mnist_cnn'
keras.models.save_model(model, os.path.join(path_wd, model_name + '.h5'))

# SNN TOOLBOX CONFIGURATION #
#############################

reset_mode = 'soft'

# Create a config file with experimental setup for SNN Toolbox.
configparser = import_configparser()
config = configparser.ConfigParser()

config['paths'] = {
    'path_wd': path_wd,             # Path to model.
    'dataset_path': path_wd,        # Path to dataset.
    'filename_ann': model_name      # Name of input model.
}

config['tools'] = {
    'evaluate_ann': True,           # Test ANN on dataset before conversion.
    'normalize': False              # Not needed for Loihi backend.
}

config['simulation'] = {
    'simulator': 'loihi',           # Chooses execution backend of SNN toolbox.
    'duration': 100,                # Number of time steps to run each sample.
    'num_to_test': 100,             # How many test samples to run.
    'batch_size': 20
}

config['loihi'] = {
    # Reset mode 'soft'/'hard' for reset-by-subtraction (more accurate)
    # or reset-to-zero (less resources).
    'reset_mode': reset_mode,
    # Scaling factor between max input and threshold.
    'desired_threshold_to_input_ratio': 2 ** 3 if reset_mode == 'hard' else 1,
    # Arguments passed to NxTF layer.
    'compartment_kwargs': {'biasExp': 6, 'vThMant': 512},
    'connection_kwargs': {'numWeightBits': 8, 'weightExponent': 0,
                          'numBiasBits': 12},
    'validate_partitions': False,   # Validates correctness of compiled layers.
    'save_output': False,           # Plots compiler information.
    'do_overflow_estimate': True,   # Estimate overflow of dendritic accumul.
    'normalize_thresholds': True,   # Tune thresholds to optimal dynamic range.
}

# Store config file.
config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# RUN SNN TOOLBOX #
###################

main(config_filepath)
