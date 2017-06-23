# coding=utf-8

"""
    Train a simple deep NN on the MNIST dataset.

    Get to 98.30% test accuracy after 20 epochs
    (there is *a lot* of margin for parameter tuning).
    2 seconds per epoch on a GRID K520 GPU.
"""

from __future__ import absolute_import
from __future__ import print_function

from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

from snntoolbox.simulation.plotting import plot_history

batch_size = 128
nb_classes = 10
nb_epoch = 50

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(128, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_history(history)

filename = '{:2.2f}'.format(score[1] * 100)
open(filename + '.json', 'w').write(model.to_json())
model.save_weights(filename + '.h5', overwrite=True)
