# coding=utf-8

"""Train a small CNN using frames generated from the DVS predator-prey data set.
"""

import os
import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GaussianNoise
from keras.callbacks import ModelCheckpoint, TensorBoard

data_path_base = '/home/rbodo/.snntoolbox/Datasets/predator_prey/npz/dvs/' \
                 'rectified_sum'
model_path_base = '/home/rbodo/.snntoolbox/data/predator_prey'

kwargs = {'input_shape': (36, 36, 1)}

activity_regularizer1 = keras.regularizers.l2(0.0001)
activity_regularizer2 = keras.regularizers.l2(0.001)
kernel_regularizer = keras.regularizers.l2(0.001)
use_bias = True

# Best settings
# data_path = os.path.join(data_path_base, 'maxpool_subsampling')
# log_path = os.path.join(model_path_base, 'logs', 'bias')

# No maxpool_subsampling
# data_path = os.path.join(data_path_base, '')
# log_path = os.path.join(model_path_base, 'logs', 'sum_subsampling')

# No 3-sigma standardization
# data_path = os.path.join(data_path_base, 'maxpool_subsampling', 'no_clip')
# log_path = os.path.join(model_path_base, 'logs', 'no_clip')

# Scaling before 3-sigma standardization
# data_path = os.path.join(data_path_base, 'maxpool_subsampling',
#                          'scale_before_clip')
# log_path = os.path.join(model_path_base, 'logs', 'scale_before_clip')

# Additive Gaussian noise
data_path = os.path.join(data_path_base, 'maxpool_subsampling')
log_path = os.path.join(model_path_base, 'logs', 'gaussian_noise2')
gaussian_noise = True

x_train = np.load(os.path.join(data_path, 'x_train.npz'))['arr_0']
x_test = np.load(os.path.join(data_path, 'x_test.npz'))['arr_0']
y_train = np.load(os.path.join(data_path, 'y_train.npz'))['arr_0']
y_test = np.load(os.path.join(data_path, 'y_test.npz'))['arr_0']

model = keras.models.Sequential()
if gaussian_noise:
    model.add(GaussianNoise(0.0001, **kwargs))
    kwargs = {}
model.add(Conv2D(4, (5, 5), use_bias=use_bias, activation='relu',
                 activity_regularizer=None,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=kernel_regularizer, **kwargs))
model.add(MaxPooling2D())
model.add(Conv2D(4, (5, 5), use_bias=use_bias, activation='relu',
                 activity_regularizer=activity_regularizer1,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=kernel_regularizer))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(40, use_bias=use_bias, activation='relu',
                activity_regularizer=activity_regularizer2,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=kernel_regularizer))
model.add(Dense(4, use_bias=use_bias, activation='softmax',
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=kernel_regularizer))

model.compile('adam', 'categorical_crossentropy', ['accuracy'])

checkpoint = ModelCheckpoint(os.path.join(
    log_path, '{epoch:02d}-{val_acc:.2f}.h5'), 'val_acc', True)
model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test),
          callbacks=[checkpoint, TensorBoard(log_path, 1)])
