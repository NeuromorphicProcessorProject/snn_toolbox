# coding=utf-8

"""

DoReFa-Net version of AlexNet:

https://github.com/ppwwyyxx/tensorpack/tree/master/examples/DoReFa-Net

"""

from __future__ import absolute_import
from __future__ import print_function

import os
import json
import keras
import numpy as np
from functools import partial
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


def quantize(x, k=2):

    n = float(2**k - 1)

    return keras.backend.round(keras.backend.clip(x, 0, 1) * n) / n


def preprocess_input(x):
    """Zero-center by mean pixel."""

    x[0, :, :] -= 103.939
    x[1, :, :] -= 116.779
    x[2, :, :] -= 123.68

    return x


batch_size = 10
target_size = (224, 224)

dataset_path = '/home/rbodo/.snntoolbox/Datasets/imagenet/validation'
class_idx_path = os.path.join(dataset_path, '..',
                              'imagenet_class_index_1000.json')
path_wd = '/home/rbodo/.snntoolbox/data/imagenet/alexnet/dorefa'

model = Sequential()

model.add(Conv2D(96, (12, 12), strides=(4, 4), use_bias=False,
                 activation=quantize, padding='same',
                 input_shape=(3, target_size[0], target_size[1])))

model.add(Conv2D(256, (5, 5), padding='same', use_bias=False))
model.add(BatchNormalization(axis=1))
model.add(Activation(quantize))

model.add(MaxPooling2D((3, 3), (2, 2), 'same'))

model.add(Conv2D(384, (3, 3), use_bias=False))
model.add(BatchNormalization(axis=1))
model.add(Activation(quantize))

model.add(MaxPooling2D((3, 3), (2, 2), 'same'))

model.add(Conv2D(384, (3, 3), use_bias=False, padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation(quantize))

model.add(Conv2D(256, (3, 3), use_bias=False, padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation(quantize))

model.add(MaxPooling2D((3, 3), (2, 2)))

model.add(Flatten())

model.add(Dense(4096, use_bias=False))
model.add(BatchNormalization(axis=1))
model.add(Activation(quantize))

model.add(Dense(4096, use_bias=False))
model.add(BatchNormalization(axis=1))
model.add(Activation(partial(keras.backend.clip, min_value=0, max_value=1)))

model.add(Dense(1000))
model.add(Activation('softmax'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

# noinspection PyTypeChecker
parameters = dict(np.load('/home/rbodo/.snntoolbox/data/imagenet/alexnet/dorefa'
                          '/AlexNet-126.npy', encoding='latin1').item())

# TODO: May have to quantize weights too (but not first or last layer!)
print([(layer.name, layer.output_shape) for layer in model.layers])

model.layers[0].set_weights([parameters['conv0/W']])

model.layers[1].set_weights([np.concatenate([parameters['conv1/W'],
                                             parameters['conv1/W']], 2)])
model.layers[2].set_weights([parameters['bn1/gamma'],
                             parameters['bn1/beta'],
                             parameters['bn1/mean/EMA'],
                             parameters['bn1/variance/EMA']])

model.layers[5].set_weights([parameters['conv2/W']])
model.layers[6].set_weights([parameters['bn2/gamma'],
                             parameters['bn2/beta'],
                             parameters['bn2/mean/EMA'],
                             parameters['bn2/variance/EMA']])

model.layers[9].set_weights([np.concatenate([parameters['conv3/W'],
                                             parameters['conv3/W']], 2)])
model.layers[10].set_weights([parameters['bn3/gamma'],
                              parameters['bn3/beta'],
                              parameters['bn3/mean/EMA'],
                              parameters['bn3/variance/EMA']])

model.layers[12].set_weights([np.concatenate([parameters['conv4/W'],
                                             parameters['conv4/W']], 2)])
model.layers[13].set_weights([parameters['bn4/gamma'],
                              parameters['bn4/beta'],
                              parameters['bn4/mean/EMA'],
                              parameters['bn4/variance/EMA']])

model.layers[17].set_weights([parameters['fc0/W']])
model.layers[18].set_weights([parameters['bnfc0/gamma'],
                              parameters['bnfc0/beta'],
                              parameters['bnfc0/mean/EMA'],
                              parameters['bnfc0/variance/EMA']])

model.layers[20].set_weights([parameters['fc1/W']])
model.layers[21].set_weights([parameters['bnfc1/gamma'],
                              parameters['bnfc1/beta'],
                              parameters['bnfc1/mean/EMA'],
                              parameters['bnfc1/variance/EMA']])

model.layers[23].set_weights([parameters['fct/W'], parameters['fct/b']])

# Data set
class_idx = json.load(open(class_idx_path, "r"))
classes = [class_idx[str(idx)][0] for idx in range(len(class_idx))]

# First scale down, then take center crop, then subtract per-pixel mean.
# When to divide by 255?

datagen = ImageDataGenerator(rescale=1/255.,
                             preprocessing_function=preprocess_input)
dataflow = datagen.flow_from_directory(dataset_path, target_size,
                                       classes=classes, batch_size=batch_size)
print("Testing...")
score = model.evaluate_generator(dataflow, steps=10)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# model.save('{:2.2f}.h5'.format(score[1]*100))
