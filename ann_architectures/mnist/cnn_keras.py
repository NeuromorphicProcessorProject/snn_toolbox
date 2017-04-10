# coding=utf-8

"""
    Train a simple convnet on the MNIST dataset.

    Run on GPU:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

    Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin
    for parameter tuning).

    16 seconds per epoch on a GRID K520 GPU.
"""

from __future__ import absolute_import
from __future__ import print_function

from keras.datasets import mnist as dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.utils.np_utils import to_categorical

from snntoolbox.io_utils.plotting import plot_history


batch_size = 128
nb_classes = 10
nb_epoch = 30

# input image dimensions
img_rows, img_cols = 28, 28
chnls = 1

(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train = x_train.reshape(x_train.shape[0], chnls, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], chnls, img_rows, img_cols)
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()

model.add(Conv2D(16, 5, 5, input_shape=(chnls, img_rows, img_cols),
                        strides=(2, 2)))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Conv2D(32, 3, 3))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(800))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(10, activation='softmax'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_history(history)

model.save('{:2.2f}.h5'.format(score[1]*100))
