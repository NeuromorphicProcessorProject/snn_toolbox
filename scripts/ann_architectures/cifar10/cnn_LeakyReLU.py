# coding=utf-8

"""
Train a (fairly simple) deep CNN on the CIFAR10 small images dataset.

GPU run command:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

Gets to about 0.5 test loss or 83% accuracy after 65 epochs.

"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.datasets import cifar10
from keras.layers import BatchNormalization, MaxPooling2D
from keras.layers import Dense, Flatten, Conv2D, LeakyReLU, Activation
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

batch_size = 32
reg = None  # keras.regularizers.l2(0.0001)

# Data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


def get_model(alpha=0, weights=None):
    keras.backend.clear_session()

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, 32, 32),
                     activity_regularizer=reg))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(alpha))
    model.add(Conv2D(32, (3, 3), padding='same', activity_regularizer=reg))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), activity_regularizer=reg))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(alpha))
    model.add(Conv2D(64, (3, 3), activity_regularizer=reg))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(512, activity_regularizer=reg))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    model.add(Dense(10, activity_regularizer=reg))
    model.add(Activation('softmax'))

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    if weights is not None:
        model.set_weights(weights)

    return model


# Whether to apply global contrast normalization and ZCA whitening
gcn = False
zca = False

traingen = ImageDataGenerator(rescale=1./255, featurewise_center=gcn,
                              featurewise_std_normalization=gcn,
                              zca_whitening=zca, horizontal_flip=True,
                              rotation_range=10, width_shift_range=0.1,
                              height_shift_range=0.1)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
if zca:
    traingen.fit(x_train/255.)

trainflow = traingen.flow(x_train, y_train, batch_size)

testgen = ImageDataGenerator(rescale=1./255, featurewise_center=gcn,
                             featurewise_std_normalization=gcn,
                             zca_whitening=zca)
if zca:
    testgen.fit(x_test/255.)

testflow = testgen.flow(x_test, y_test, batch_size)

path = '/home/rbodo/.snntoolbox/data/cifar10/linear'
checkpointer = ModelCheckpoint('cnn_BN.{epoch:02d}-{val_acc:.2f}.h5',
                               verbose=1, save_best_only=True)

import keras
model_pretrained = keras.models.load_model('/home/rbodo/.snntoolbox/data/'
                                           'cifar10/87.86/87.86.h5')
weights = model_pretrained.get_weights()

model = get_model(0, weights)

score = model.evaluate_generator(testflow, len(x_test)/batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model = get_model(1, weights)

score = model.evaluate_generator(testflow, len(x_test)/batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])

nb_epoch = 1
alpha = 0
steps = 10
alphas = np.linspace(0, 1, steps)

for alpha in alphas:
    tensorboard = TensorBoard(path + '/logs/refined_' + str(alpha), 5,
                              write_grads=True)
    callbacks = [checkpointer, tensorboard]

    model = get_model(alpha, model.get_weights())

    validation_data = testflow.next()
    history = model.fit_generator(trainflow, len(x_train)/batch_size, nb_epoch,
                                  verbose=1, callbacks=callbacks,
                                  validation_data=validation_data)  # testflow
                                  # validation_steps=len(x_test)/batch_size)

    score = model.evaluate_generator(testflow, len(x_test)/batch_size)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

model.save(path + '/{:2.2f}.h5'.format(score[1]*100))
