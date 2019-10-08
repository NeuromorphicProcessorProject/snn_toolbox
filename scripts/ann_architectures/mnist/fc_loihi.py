# coding=utf-8

"""
    Train a simple MLP on the MNIST dataset.
"""

from __future__ import absolute_import
from __future__ import print_function

from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist as dataset
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

from snntoolbox.simulation.plotting import plot_history
from skimage.measure import block_reduce
import numpy as np

batch_size = 128
nb_classes = 10
nb_epoch = 10

(x_train, y_train), (x_test, y_test) = dataset.load_data()
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

subsample_shift = 4

x_train = block_reduce(x_train, (1, subsample_shift, subsample_shift), np.mean)
x_test = block_reduce(x_test, (1, subsample_shift, subsample_shift), np.mean)

x_train /= 255.
x_test /= 255.

x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))

path = '/home/brueckau/Downloads/'
np.savez_compressed(path + 'x_test', x_test)
np.savez_compressed(path + 'x_train', x_train)

model = Sequential()

model.add(Dense(800, input_shape=(x_train.shape[-1],), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

model.summary()

checkpointer = ModelCheckpoint(filepath=path+'fc.{epoch:02d}-{val_acc:.2f}.h5',
                               verbose=1, save_best_only=True)

history = model.fit(x_train, y_train, batch_size, nb_epoch,
                    callbacks=[checkpointer],
                    validation_data=(x_test, y_test))
plot_history(history)
