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
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

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
model.add(Dense(600, batch_input_shape=(batch_size, 784), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
# model.add(Dense(128, input_shape=(784,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(10))
# model.add(Activation('softmax'))

optimizer = Adam()
model.compile(optimizer, 'categorical_crossentropy', ['accuracy'])

checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_acc:.2f}.h5', 'val_acc')
history = model.fit(X_train, Y_train, batch_size, nb_epoch,
                    validation_data=(X_test, Y_test), callbacks=[checkpoint])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_history(history)

model.save('{:2.2f}.h5'.format(score[1]*100))
