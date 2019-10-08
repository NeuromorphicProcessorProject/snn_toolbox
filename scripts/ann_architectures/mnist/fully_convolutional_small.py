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
from keras.datasets import mnist as dataset
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Conv2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from skimage.measure import block_reduce

from snntoolbox.simulation.plotting import plot_history

save_dataset = False

batch_size = 32
epochs = 40
nb_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28
chnls = 1

(x_train, y_train), (x_test, y_test) = dataset.load_data()
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

subsample_shift = 4

x_train = block_reduce(x_train, (1, subsample_shift, subsample_shift), np.mean)
x_test = block_reduce(x_test, (1, subsample_shift, subsample_shift), np.mean)

x_train /= 255.
x_test /= 255.

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

path = '/home/brueckau/Downloads'
if save_dataset:
    np.savez_compressed(path + '/x_test', x_test)
    np.savez_compressed(path + '/x_train', x_train)
    np.savez_compressed(path + '/y_test', y_test)

img_rows = int(img_rows / subsample_shift)
img_cols = int(img_cols / subsample_shift)

nonlinearity = 'relu'
use_bias = False

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=(img_cols, img_rows, chnls),
                 activation=nonlinearity, use_bias=use_bias))
# model.add(Dropout(0.1))

model.add(Conv2D(32, (3, 3), activation=nonlinearity, use_bias=use_bias))
# model.add(Dropout(0.1))

model.add(Conv2D(10, (3, 3), activation='softmax', use_bias=use_bias))

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
