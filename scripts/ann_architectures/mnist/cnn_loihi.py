# coding=utf-8

"""
    Train a simple convnet on the MNIST dataset.

    Gets to 97.59% val acc after 10 epochs (which takes about 5 min on vlab).
    Not yet converged at this point, and not overfitting.
"""

from __future__ import absolute_import
from __future__ import print_function

from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist as dataset
from keras.layers import BatchNormalization, Activation, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from snntoolbox.simulation.plotting import plot_history
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import numpy as np

batch_size = 128
nb_classes = 10
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 28, 28
chnls = 1

(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, chnls)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, chnls)
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

subsample_shift = 4

x_train = block_reduce(x_train, (1, subsample_shift, subsample_shift, 1), np.mean)
x_test = block_reduce(x_test, (1, subsample_shift, subsample_shift, 1), np.mean)

img_rows = int(img_rows / subsample_shift)
img_cols = int(img_cols / subsample_shift)

model = Sequential()

model.add(Conv2D(200, (3, 3), input_shape=(img_rows, img_cols, chnls)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Conv2D(100, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Conv2D(100, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(10, activation='softmax'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

model.summary()

traingen = ImageDataGenerator(rescale=1./255)
trainflow = traingen.flow(x_train, y_train, batch_size=batch_size)

testgen = ImageDataGenerator(rescale=1./255)
testflow = testgen.flow(x_test, y_test, batch_size=batch_size)

checkpointer = ModelCheckpoint(filepath='cnn.{epoch:02d}-{val_acc:.2f}.h5',
                               verbose=1, save_best_only=True)

history = model.fit_generator(trainflow, len(x_train) / batch_size, nb_epoch,
                              callbacks=[checkpointer],
                              validation_data=testflow,
                              validation_steps=len(x_test) / batch_size)
plot_history(history)
