# coding=utf-8

"""
    Train a simple convnet on the MNIST dataset, without biases.

"""

from __future__ import absolute_import
from __future__ import print_function

from keras.datasets import mnist as dataset
from keras.layers import Dense, Flatten, Conv2D
from keras.models import Sequential
from keras.utils import np_utils

from snntoolbox.simulation.plotting import plot_history

nb_classes = 10
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 28, 28
# color channels
chnls = 1

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = dataset.load_data()

X_train = X_train.reshape(X_train.shape[0], chnls, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], chnls, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()

model.add(Conv2D(32, 3, 3, activation='relu',
                        input_shape=(chnls, img_rows, img_cols)))
model.add(Conv2D(32, 1, 1, activation='relu'))
model.add(Flatten())
model.add(Dense(nb_classes, activation='softmax'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print("Training...")
history = model.fit(X_train, Y_train, nb_epoch, verbose=0,
                    validation_data=(X_test, Y_test))
print("Testing...")
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_history(history)

model.save('{:2.2f}.h5'.format(score[1]*100))
