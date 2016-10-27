# -*- coding: utf-8 -*-

"""
VGG16 model for Keras, from https://github.com/fchollet/deep-learning-models.

Reference:

[Very Deep Convolutional Networks for Large-Scale Image Recognition]
(https://arxiv.org/abs/1409.1556)

"""

from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from ann_architectures.imagenet.utils import decode_predictions
from ann_architectures.imagenet.utils import preprocess_input


def get_vgg16():
    model = Sequential()

    # Block 1
    model.add(Convolution2D(64, 3, 3, border_mode='same', name='block1_conv1',
                            input_shape=(3, 224, 224), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', name='block1_conv2',
                            activation='relu'))
    model.add(MaxPooling2D(name='block1_pool'))

    # Block 2
    model.add(Convolution2D(128, 3, 3, border_mode='same',
                            name='block2_conv1', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same',
                            name='block2_conv2', activation='relu'))
    model.add(MaxPooling2D(name='block2_pool'))

    # Block 3
    model.add(Convolution2D(256, 3, 3, border_mode='same',
                            name='block3_conv1', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same',
                            name='block3_conv2', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same',
                            name='block3_conv3', activation='relu'))
    model.add(MaxPooling2D(name='block3_pool'))

    # Block 4
    model.add(Convolution2D(512, 3, 3, border_mode='same',
                            name='block4_conv1', activation='relu'))
    model.add(Convolution2D(512, 3, 3, border_mode='same',
                            name='block4_conv2', activation='relu'))
    model.add(Convolution2D(512, 3, 3, border_mode='same',
                            name='block4_conv3', activation='relu'))
    model.add(MaxPooling2D(name='block4_pool'))

    # Block 5
    model.add(Convolution2D(512, 3, 3, border_mode='same',
                            name='block5_conv1', activation='relu'))
    model.add(Convolution2D(512, 3, 3, border_mode='same',
                            name='block5_conv2', activation='relu'))
    model.add(Convolution2D(512, 3, 3, border_mode='same',
                            name='block5_conv3', activation='relu'))
    model.add(MaxPooling2D(name='block5_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, name='fc1', activation='relu'))
    model.add(Dense(4096, name='fc2', activation='relu'))
    model.add(Dense(1000, name='predictions', activation='softmax'))

    return model


if __name__ == '__main__':

    model = get_vgg16()
    model.compile('sgd', 'categorical_crossentropy', ['accuracy'])

#    # load weights
#    weights_path = 'https://github.com/fchollet/deep-learning-models/' + \
#        'releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
#    weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels.h5',
#                            weights_path, cache_subdir='models')
    weights_path = '/home/rbodo/.snntoolbox/data/imagenet/vgg16/INI/70.88.h5'
    model.load_weights(weights_path)
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))

    filename = 'vgg16_no_classifier'
    open(filename + '.json', 'w').write(model.to_json())
    model.save_weights(filename + '.h5', overwrite=True)
