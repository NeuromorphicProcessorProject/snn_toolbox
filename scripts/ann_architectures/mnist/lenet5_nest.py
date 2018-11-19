# coding=utf-8

"""
LeNet for MNIST

Modified to improve SNN conversion and simulation on NEST:
- No biases
- Average instead of MaxPooling
(- No softmax in output layer.)
"""

import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

batch_size = 32
epochs = 10

input_shape = [28, 28, 1]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = \
    x_train.reshape([x_train.shape[0]] + input_shape).astype('float32') / 255.
x_test = \
    x_test.reshape([x_test.shape[0]] + input_shape).astype('float32') / 255.
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

nonlinearity = 'relu'
use_bias = False

model = Sequential()

model.add(Conv2D(6, (5, 5), input_shape=input_shape, activation=nonlinearity,
                 use_bias=use_bias))
model.add(AveragePooling2D())

model.add(Conv2D(16, (5, 5), activation=nonlinearity, use_bias=use_bias))
model.add(AveragePooling2D())
model.add(Dropout(0.5))

model.add(Conv2D(120, (5, 5), padding='same', activation=nonlinearity,
                 use_bias=use_bias))

model.add(Flatten())
model.add(Dense(84, activation=nonlinearity, use_bias=use_bias))
model.add(Dense(10, activation='softmax', use_bias=use_bias))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

path = '/home/rbodo/Downloads'

checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_acc:.2f}.h5', 'val_acc',
                             save_best_only=True)
callbacks = [checkpoint]
model.fit(x_train, y_train, batch_size, epochs,
          validation_data=(x_test, y_test), callbacks=callbacks)

score = model.evaluate(x_test, y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save(os.path.join(path, '{:2.2f}.h5'.format(score[1]*100)))
