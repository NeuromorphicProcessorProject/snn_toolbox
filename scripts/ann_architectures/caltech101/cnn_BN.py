# coding=utf-8

"""
Train a CNN on a data augmented version of the Caltech101 images dataset.
"""

from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from snntoolbox.simulation.plotting import plot_history

batch_size = 64
nb_epoch = 100
nb_classes = 102
nb_samples = 9144

# Shape of the image
img_cols, img_rows = 240, 180
# The caltech101 images are RGB
img_channels = 3

datapath = '/home/rbodo/.snntoolbox/Datasets/caltech101/'

init = 'he_normal'
reg = l2(5e-4)  # Weight regularization value for l2

model = Sequential()
model.add(Conv2D(128, 5, 5, strides=(2, 2), init=init,
                        W_regularizer=reg,
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(AveragePooling2D())

model.add(Conv2D(256, 3, 3, init=init, W_regularizer=reg))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(AveragePooling2D())

model.add(Conv2D(512, 3, 3, init=init, W_regularizer=reg))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(AveragePooling2D())

model.add(Flatten())

model.add(Dense(1024, init=init, W_regularizer=reg))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(nb_classes, init=init, W_regularizer=reg))
model.add(Activation('softmax'))

sgd = SGD(lr=0.005, decay=5e-4, momentum=0.9, nesterov=True)
model.compile(sgd, 'categorical_crossentropy', metrics=['accuracy'])

target_size = (img_rows, img_cols)
flow_kwargs = {'directory': os.path.join(datapath, 'original'),
               'target_size': target_size, 'batch_size': batch_size}

# Train using preprocessing and realtime data augmentation
traingen = ImageDataGenerator(rescale=1./255,
                              featurewise_center=True,
                              featurewise_std_normalization=True,
                              rotation_range=30,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=0.1,
                              horizontal_flip=True)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
kwargs = flow_kwargs.copy()
kwargs['batch_size'] = nb_samples
X_orig = ImageDataGenerator(rescale=1./255).flow_from_directory(
    **kwargs).next()[0]
traingen.fit(X_orig)

trainflow = traingen.flow_from_directory(**flow_kwargs)

testgen = ImageDataGenerator(rescale=1./255,
                             featurewise_center=True,
                             featurewise_std_normalization=True)
testgen.fit(X_orig)

testflow = testgen.flow_from_directory(**flow_kwargs)

history = model.fit_generator(
    trainflow, samples_per_epoch=nb_samples, verbose=2, nb_epoch=nb_epoch,
    validation_data=testflow, nb_val_samples=int(nb_samples/10))

plot_history(history)

# Test
score = model.evaluate_generator(testflow, val_samples=nb_samples)

print("Test score:", score[0])
print("Test accuracy:", score[1])

batch_idx = 0
nb_batches = int(nb_samples/batch_size)
count = np.zeros(nb_classes)
match = np.zeros(nb_classes)
for X, Y in testflow:
    batch_idx += 1
    truth = np.argmax(Y, axis=1)
    predictions = np.argmax(model.predict_on_batch(X), axis=1)
    for gt, p in zip(truth, predictions):
        count[gt] += 1
        if gt == p:
            match[gt] += 1
    if batch_idx > nb_batches:
        break
class_acc = match / count
print("Class accuracies: {:2.2%}".format(class_acc))
avg_acc = np.mean(class_acc)
print("Accuracy averaged over classes: {:2.2%}".format(avg_acc))

model.save('{:2.2f}.h5'.format(avg_acc*100))
