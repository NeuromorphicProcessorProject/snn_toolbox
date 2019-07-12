# coding=utf-8

"""
CNN for MNIST

Modified to improve SNN conversion for running on Loihi:
- No biases
- Convolution with stride 2 instead of MaxPooling
(- No softmax in output layer.)
"""

import os
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Conv2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from snntoolbox.simulation.plotting import plot_history

save_dataset = False

batch_size = 32
epochs = 20

input_shape = [28, 28, 1]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = \
    x_train.reshape([x_train.shape[0]] + input_shape).astype('float32') / 255.
x_test = \
    x_test.reshape([x_test.shape[0]] + input_shape).astype('float32') / 255.
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

path = '/home/brueckau/Downloads'
if save_dataset:
    np.savez_compressed(path + '/x_test', x_test)
    np.savez_compressed(path + '/x_train', x_train)
    np.savez_compressed(path + '/y_test', y_test)

nonlinearity = 'relu'
use_bias = False

model = Sequential()

model.add(Conv2D(16, (5, 5), strides=(2, 2), input_shape=input_shape,
                 activation=nonlinearity, use_bias=use_bias))
model.add(Dropout(0.1))

model.add(Conv2D(32, (3, 3), activation=nonlinearity, use_bias=use_bias))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), strides=(2, 2), activation=nonlinearity,
                 use_bias=use_bias))
model.add(Dropout(0.1))

model.add(Conv2D(10, (4, 4), activation='softmax', use_bias=use_bias))

model.add(Flatten())

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

model.summary()

checkpointer = ModelCheckpoint(path+'/cnn.{epoch:02d}-{val_acc:.2f}.h5',
                               'val_acc', verbose=1, save_best_only=True)

history = model.fit(x_train, y_train, batch_size, epochs, verbose=2,
                    validation_data=(x_test, y_test), callbacks=[checkpointer])

score = model.evaluate(x_test, y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save(os.path.join(path, '{:2.2f}.h5'.format(score[1]*100)))

plot_history(history)
