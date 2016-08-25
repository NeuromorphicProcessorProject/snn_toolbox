from __future__ import absolute_import
from __future__ import print_function

from keras.datasets import mnist as dataset
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.utils import np_utils
from keras.constraints import maxnorm

from snntoolbox.io_utils.plotting import plot_history


"""
    Train a simple convnet on the MNIST dataset, without biases.

"""

batch_size = 128
nb_classes = 10
nb_epoch = 40

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
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

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, b_constraint=maxnorm(0),
                        input_shape=(chnls, img_rows, img_cols)))
model.add(Activation('relu'))
# model.add(AveragePooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Flatten())
model.add(Dense(nb_classes, b_constraint=maxnorm(0)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_history(history)

filename = '{:2.2f}'.format(score[1] * 100)
open(filename + '.json', 'w').write(model.to_json())
model.save_weights(filename + '.h5', overwrite=True)
