from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import cifar10

from snntoolbox.io_utils.plotting import plot_history

'''
    Train a simple deep NN on the cifar10 dataset.
'''

batch_size = 128
nb_classes = 10
nb_epoch = 50

data_augmentation = True

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
input_length = int(np.prod(X_train.shape[1:]))
X_train = X_train.reshape(X_train.shape[0], input_length)
X_test = X_test.reshape(X_test.shape[0], input_length)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(input_length,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, batch_size=batch_size)

print('Test score:', score[0])
print('Test accuracy:', score[1])
plot_history(history)

filename = '{:2.2f}'.format(score[1] * 100)
open(filename + '.json', 'w').write(model.to_json())
model.save_weights(filename + '.h5', overwrite=True)
