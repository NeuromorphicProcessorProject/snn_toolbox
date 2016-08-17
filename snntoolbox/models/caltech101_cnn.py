'''
Train a CNN on a data augmented version of the Caltech101 images dataset.

'''

from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from snntoolbox.io_utils.plotting import plot_history

batch_size = 64
nb_epoch = 2
nb_classes = 102
nb_samples = 9144

# Shape of the image
img_cols, img_rows = 240, 180
# The caltech101 images are RGB
img_channels = 3

datapath = '/home/rbodo/.snntoolbox/datasets/caltech101/'

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

# Train using preprocessing and realtime data augmentation
traingen = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=90,
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images

trainflow = traingen.flow_from_directory(
    os.path.join(datapath, 'original'), target_size=(img_rows, img_cols),
    batch_size=batch_size)  # save_to_dir=os.path.join(datapath, 'processed'))

testgen = ImageDataGenerator(rescale=1./255)

testflow = testgen.flow_from_directory(
    os.path.join(datapath, 'original'), target_size=(img_rows, img_cols),
    batch_size=batch_size)

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
print("Class accuracies: {}".format(class_acc))
avg_acc = np.mean(class_acc)
print("Accuracy averaged over classes: {}".format(avg_acc))

# Save
filename = '/home/rbodo/{:2.2f}'.format(avg_acc * 100)
open(filename + '.json', 'w').write(model.to_json())
model.save_weights(filename + '.h5', overwrite=True)
