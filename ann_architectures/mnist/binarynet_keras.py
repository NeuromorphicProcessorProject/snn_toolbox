from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import MaxPooling2D, Dropout, Flatten
from ann_architectures.BinaryConnect.binary_net_keras import Convolution2D
from ann_architectures.BinaryConnect.binary_net_keras import Dense
from ann_architectures.BinaryConnect.common import binary_sigmoid_unit
from snntoolbox.io_utils.common import load_dataset

from snntoolbox.io_utils.plotting import plot_history


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
nb_epoch = 100

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

activation = binary_sigmoid_unit

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        input_shape=(chnls, img_rows, img_cols),
                        activation=activation))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation=activation))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

path_to_dataset = '/home/rbodo/.snntoolbox/Datasets/mnist/cnn'
X_train = load_dataset(path_to_dataset, 'X_norm.npz')
Y_train = load_dataset(path_to_dataset, 'Y_train.npz')
X_test = load_dataset(path_to_dataset, 'X_test.npz')
Y_test = load_dataset(path_to_dataset, 'Y_test.npz')

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
#                    callbacks=[early_stopping])
score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_history(history)

model.save('{:2.2f}.h5'.format(score[1]*100))
