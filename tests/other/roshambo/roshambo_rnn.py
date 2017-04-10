# -*- coding: utf-8 -*-

"""Train hybrid RNN-CNN on DVS data for Roshambo task."""

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, GRU, Reshape

dataset_path = '/home/rbodo/.snntoolbox/Datasets/roshambo'
x_train = np.load(os.path.join(dataset_path, 'x_test.npz'))['arr_0']
y_train = np.load(os.path.join(dataset_path, 'y_test.npz'))['arr_0']

batch_size = 30
epochs = 2

img_shape = x_train[0].shape
input_dim = int(np.product(img_shape))
x_train_flat = np.reshape(x_train, (-1, input_dim))
x_train_t = np.stack(np.split(x_train_flat, 10), 1)
y_train_t = y_train[::10]

model = Sequential()
model.add(GRU(input_dim, input_shape=(10, input_dim), activation='relu'))
model.add(Reshape((1, 64, 64)))
model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='relu'))  # 30x30
model.add(Conv2D(16, (3, 3), strides=(2, 2), activation='relu'))  # 14x14
model.add(Conv2D(8, (3, 3), strides=(2, 2), activation='relu'))  # 6x6
model.add(Flatten())
model.add(Dense(4, activation='relu'))

model.compile('rmsprop', 'categorical_crossentropy', ['accuracy'])

model.fit(x_train_t, y_train_t, batch_size, epochs, validation_split=0.2,
          verbose=1)

score, acc = model.evaluate(x_train_t, y_train_t, batch_size)

print("\nAccuracy: {}".format(acc))
