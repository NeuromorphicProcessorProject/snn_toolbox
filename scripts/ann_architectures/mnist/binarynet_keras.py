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

from keras.layers import MaxPooling2D, Dropout, Flatten
from keras.models import Sequential

from scripts.ann_architectures.BinaryConnect import Conv2D
from scripts.ann_architectures.BinaryConnect import Dense
from scripts.ann_architectures.BinaryConnect import binary_sigmoid_unit
from snntoolbox.datasets.utils import load_npz
from snntoolbox.simulation.plotting import plot_history

batch_size = 128
nb_classes = 10
nb_epoch = 10

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

activation = binary_sigmoid_unit

model = Sequential()

model.add(Conv2D(filters, nb_conv, nb_conv,
                 input_shape=(chnls, img_rows, img_cols),
                 activation=activation))
model.add(Conv2D(filters, nb_conv, nb_conv, activation=activation))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

path_to_dataset = '/home/rbodo/.snntoolbox/Datasets/mnist/cnn'
X_train = load_npz(path_to_dataset, 'x_norm.npz')
Y_train = load_npz(path_to_dataset, 'y_train.npz')
X_test = load_npz(path_to_dataset, 'x_test.npz')
Y_test = load_npz(path_to_dataset, 'y_test.npz')

history = model.fit(X_train, Y_train, batch_size, nb_epoch,
                    validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_history(history)

model.save('{:2.2f}.h5'.format(score[1]*100))
