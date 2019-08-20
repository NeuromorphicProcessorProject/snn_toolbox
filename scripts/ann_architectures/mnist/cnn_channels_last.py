# coding=utf-8

"""LeNet for MNIST"""

import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

batch_size = 32
epochs = 3

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

use_bias = False
nonlinearity = 'relu'

model = Sequential()

model.add(Conv2D(6, (5, 5), input_shape=(28, 28, 1), activation=nonlinearity,
                 use_bias=use_bias, strides=(2, 2)))
model.add(AveragePooling2D())

model.add(Flatten())
model.add(Dense(10, activation='softmax', use_bias=use_bias))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

path = '/home/brueckau/Downloads'

checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_acc:.2f}.h5', 'val_acc')
callbacks = [checkpoint]
model.fit(x_train, y_train, batch_size, epochs,
          validation_data=(x_test, y_test), callbacks=callbacks)

score = model.evaluate(x_test, y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save(os.path.join(path, '{:2.2f}.h5'.format(score[1]*100)))
