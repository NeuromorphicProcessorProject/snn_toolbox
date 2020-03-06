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
import datetime
import numpy as np
import pandas as pd

import keras
from keras import Input, Model
from keras.layers import Conv2D, Flatten, Dense, Dropout, GaussianNoise, \
    DepthwiseConv2D
from keras.layers import BatchNormalization, Activation
from keras.utils import np_utils

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser

# Enable SLURM to run network on Loihi.
os.environ['SLURM'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

# Add path to the nxsdk_modules package.
sys.path.append('/homes/rbodo/Repositories/nxtf/nxsdk-apps')

# WORKING DIRECTORY #
#####################

# Define path where model and output files will be stored.
# The user is responsible for cleaning up this temporary directory.
path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp', str(time.time())))
os.makedirs(path_wd)

# GET DATASET #
###############

# u'Conductivitat (20 °C)', 'isf1_c', 'isf2_c', 'isf3_c', 'isf4_c', 'isf5_c',
# 'isf6_c', 'orp', 'temp']
out_meas = ['LABEL']
in_meas = ['ISF1', 'ISF2', 'ISF3', 'ISF4', 'ISF5',
           'ISF6', 'ORP', 'COND', 'TEMP']

df_all_norm = pd.read_pickle('df_realtaste_lab_filt_std_py3_allmeas.pkl')
# Labels start at 0 instead of 1
df_all_norm['LABEL'] = df_all_norm['LABEL'] - 1

n_test = 2281
n_disc = 787
n_vars = len(in_meas)
n_timesteps = 10
n_classes = 5


def create_train_test(x, y):
    return x[:-n_test], x[-n_test:-n_disc], y[:-n_test], y[-n_test:-n_disc]


df_all_5s = df_all_norm[out_meas + in_meas].copy()
y_ph_2_5s = df_all_5s[out_meas].dropna()
xl2_ph_5s = []

# u'Conductivitat (20 °C)-'+str(i*5)+'sec', 'isf1_c-'+str(i*5)+'sec',
# 'temp-'+str(i*5)+'sec'])) #[u'Conductivitat (20 °C)-'+str(i*5)+'sec',
# 'isf1_c-'+str(i*5)+'sec', 'isf2_c-'+str(i*5)+'sec', 'isf3_c-'+str(i*5)+'sec',
# 'isf4_c-'+str(i*5)+'sec', 'isf5_c-'+str(i*5)+'sec', 'isf6_c-'+str(i*5)+'sec',
# 'orp-'+str(i*5)+'sec', 'temp-'+str(i*5)+'sec']))
for i in reversed(range(n_timesteps)):
    xl2_ph_5s.append(pd.DataFrame(df_all_norm.reindex(
        y_ph_2_5s.index - datetime.timedelta(seconds=i))[in_meas].values,
                                  columns=[m + '-{}sec'.format(i)
                                           for m in in_meas]))

x_ph_5s = pd.concat(xl2_ph_5s, axis=1)
x_ph_5s.index = y_ph_2_5s.index
x_ph_5s = x_ph_5s.dropna()
y_ph_5s = y_ph_2_5s
y_ph_5s = y_ph_5s.reindex(x_ph_5s.index)

# scaler.fit_transform(pylab.expand_dims(y_ph_5s.values,axis=1))
y_ph_5s_s = y_ph_5s.values.astype('float32')
# scaler.fit_transform(x_ph_5s.values)
x_ph_5s_s = x_ph_5s.values.astype('float32')

x_ph_5s_r = np.reshape(x_ph_5s_s, (-1, n_timesteps, 1, n_vars))
x_ph_5s_r -= np.min(x_ph_5s_r)
x_ph_5s_r /= np.max(x_ph_5s_r)
y_ph_5s_r = np_utils.to_categorical(y_ph_5s_s, n_classes)

p = np.random.permutation(len(x_ph_5s_r))
x_ph_5s_r = x_ph_5s_r[p]
y_ph_5s_r = y_ph_5s_r[p]

x_ph_5s_r = np.repeat(x_ph_5s_r, 3, 2)

train_x, test_x, train_y, test_y = create_train_test(x_ph_5s_r, y_ph_5s_r)

np.savez_compressed(os.path.join(path_wd, 'x_test'), test_x)
np.savez_compressed(os.path.join(path_wd, 'y_test'), test_y)
np.savez_compressed(os.path.join(path_wd, 'x_norm'), train_x)

# CREATE ANN #
##############


noise_std = 0.8
n_filters = 1
kernel_shape = (3, 3)
kernel_init = 'he_uniform'
pad = 'same'
dropout = 1e-2
use_bias = True
use_bn = True
use_dw = False

input_layer = Input(shape=(n_timesteps, 3, n_vars))
if use_dw:
    layer = DepthwiseConv2D(kernel_size=kernel_shape, padding=pad,
                            # depth_multiplier=n_filters,
                            kernel_initializer=kernel_init,
                            use_bias=use_bias)(input_layer)
else:
    layer = Conv2D(32, kernel_shape, padding=pad,
                   use_bias=use_bias)(input_layer)
if use_bn:
    layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = Dropout(dropout)(layer)
if use_dw:
    layer = DepthwiseConv2D(kernel_size=kernel_shape, padding=pad,
                            kernel_initializer=kernel_init,
                            dilation_rate=2, use_bias=use_bias)(layer)
else:
    layer = Conv2D(64, kernel_shape, padding=pad, use_bias=use_bias)(layer)
if use_bn:
    layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = Dropout(dropout)(layer)
layer = Conv2D(50, 1, use_bias=use_bias)(layer)
if use_bn:
    layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = Flatten()(layer)
layer = GaussianNoise(noise_std)(layer)
layer = Dense(n_classes, activation='softmax', use_bias=use_bias)(layer)
model = Model(input_layer, layer)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=20,
          batch_size=64, verbose=2)

keras.models.save_model(model, os.path.join(path_wd, 'model.h5'))

# SNN TOOLBOX CONFIGURATION #
#############################

reset_mode = 'soft'

configparser = import_configparser()
config = configparser.ConfigParser()

config['paths'] = {
    'path_wd': path_wd,
    'dataset_path': path_wd,
    'filename_ann': 'model'
}

config['tools'] = {
    'evaluate_ann': True,
    'normalize': False
}

config['simulation'] = {
    'simulator': 'loihi',
    'duration': 512,
    'num_to_test': 1494,
    'batch_size': 1,
}

config['loihi'] = {
    'reset_mode': reset_mode,
    'desired_threshold_to_input_ratio': 2 ** 3 if reset_mode == 'hard' else 1,
    'compartment_kwargs': {'biasExp': 6, 'vThMant': 512},
    'connection_kwargs': {'numWeightBits': 8, 'weightExponent': 0,
                          'numBiasBits': 12},
    'validate_partitions': False,
    'save_output': False,
    'use_reset_snip': True,
    'do_overflow_estimate': False,
    'normalize_thresholds': True,
}

# config['output'] = {
#     'plot_vars': {'all'}
# }

# Store config file.
config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# RUN SNN TOOLBOX #
###################

main(config_filepath)
