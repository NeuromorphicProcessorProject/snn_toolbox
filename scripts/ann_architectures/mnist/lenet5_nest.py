# coding=utf-8

"""
LeNet for MNIST

Modified to improve SNN conversion and simulation on NEST:
- Average instead of MaxPooling
- No softmax in output layer.
"""

import os
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import keras.backend as k

path = '/home/rbodo/Downloads'
# Using relu as output activation can sometimes cause an accuracy drop of > 10%
# depending on the random seed.
output_nonlinearity = 'relu'
loss = 'mse'  # 'categorical_crossentropy'
batch_size = 32
epochs = 10
axis = 1 if k.image_data_format() == 'channels_first' else -1
flatten_arg = None  # 'channels_last'

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train.astype('float32') / 255., axis)
x_test = np.expand_dims(x_test.astype('float32') / 255., axis)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

input_shape = x_train.shape[1:]

model = Sequential()

model.add(Conv2D(6, (5, 5), input_shape=input_shape, activation='relu'))
model.add(AveragePooling2D())

model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(AveragePooling2D())
model.add(Dropout(0.5))

model.add(Conv2D(120, (5, 5), padding='same', activation='relu'))

model.add(Flatten(flatten_arg))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation=output_nonlinearity))

model.compile('adam', loss, metrics=['accuracy'])

checkpoint = ModelCheckpoint('{epoch:02d}-{val_acc:.2f}.h5', 'val_acc',
                             save_best_only=True)
callbacks = [checkpoint]
model.fit(x_train, y_train, batch_size, epochs,
          validation_data=(x_test, y_test), callbacks=callbacks)

score = model.evaluate(x_test, y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save(os.path.join(path, '{:2.2f}.h5'.format(score[1]*100)))
