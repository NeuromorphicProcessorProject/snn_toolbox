# coding=utf-8

"""A example for MNIST CNN with larger pool size."""

from __future__ import absolute_import
from __future__ import print_function

from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils

from snntoolbox.simulation.plotting import plot_history

batch_size = 128
nb_classes = 10
nb_epoch = 80

data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()

model.add(Conv2D(32, 3, 3, padding='valid',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, 3, 3, padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta',
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
#                    callbacks=[early_stopping])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_history(history)

filename = '{:2.2f}'.format(score[1] * 100)+".cifar.larger.pool"
open(filename + '.json', 'w').write(model.to_json())
model.save_weights(filename + '.h5', overwrite=True)
