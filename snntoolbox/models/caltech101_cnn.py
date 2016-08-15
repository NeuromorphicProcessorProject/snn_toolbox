'''
Train a CNN on a data augmented version of the Caltech101 images dataset.

'''

from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD

import numpy as np
from snntoolbox.io_utils.plotting import plot_history

batch_size = 64
nb_epoch = 64
nb_classes = 102

# Shape of the image
img_rows, img_cols = 180, 240
# The caltech101 images are RGB
img_channels = 3

# Load dataset
dataset_path = '/home/rbodo/.snntoolbox/datasets/caltech101/small/caltech101'
X_train = np.load(dataset_path+'_X_train.npz')['arr_0']
X_test = np.load(dataset_path+'_X_test.npz')['arr_0']
Y_train = np.load(dataset_path+'_Y_train.npz')['arr_0']
Y_test = np.load(dataset_path+'_Y_test.npz')['arr_0']

weight_reg = 5e-4  # Weight regularization value for l2

model = Sequential()
conv1 = Convolution2D(128, 5, 5,
                      subsample=(2, 2),  # subsample = stride
                      init='he_normal',
                      W_regularizer=l2(weight_reg),
                      input_shape=(img_channels, img_rows, img_cols))
model.add(conv1)
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

conv2 = Convolution2D(256, 3, 3, init='he_normal',
                      W_regularizer=l2(weight_reg))
model.add(conv2)
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

conv3 = Convolution2D(512, 3, 3, init='he_normal',
                      W_regularizer=l2(weight_reg))
model.add(conv3)
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(1024, init='he_normal', W_regularizer=l2(weight_reg)))
model.add(BatchNormalization(mode=0))
model.add(Activation('relu'))

model.add(Dense(nb_classes, init='he_normal', W_regularizer=l2(weight_reg)))
model.add(Activation('softmax'))

sgd = SGD(lr=0.005, decay=5e-4, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy'])

# Train
history = model.fit(X_train, Y_train, batch_size=batch_size,
                    validation_data=(X_test, Y_test), nb_epoch=nb_epoch)
plot_history(history)

# Test
score = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Save
filename = '/home/rbodo/{:2.2f}'.format(score[1] * 100)
open(filename + '.json', 'w').write(model.to_json())
model.save_weights(filename + '.h5', overwrite=True)
