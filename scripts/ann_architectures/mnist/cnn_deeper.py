# coding=utf-8

"""A example for MNIST CNN with deeper layers."""

from __future__ import absolute_import
from __future__ import print_function

from keras.callbacks import EarlyStopping
from keras.datasets import mnist as dataset
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils

from snntoolbox.simulation.plotting import plot_history

"""
    Train a simple convnet on the MNIST dataset.

    Run on GPU:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

    Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin
    for parameter tuning).

    16 seconds per epoch on a GRID K520 GPU.
"""

batch_size = 128
nb_classes = 10
nb_epoch = 40

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
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

model.add(Conv2D(20, 3, 3,
                        input_shape=(chnls, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Conv2D(20, 3, 3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(20, 3, 3))
model.add(Activation('relu'))
model.add(Conv2D(20, 3, 3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta',
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
#                    callbacks=[early_stopping])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_history(history)

filename = '{:2.2f}'.format(score[1] * 100)
open(filename + '.json', 'w').write(model.to_json())
model.save_weights(filename + '.h5', overwrite=True)
