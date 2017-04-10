"""VGG16 with MaxPooling from fchollet.

Please see more details at
[here](https://github.com/fchollet/deep-learning-models)

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
# -*- coding: utf-8 -*-
from __future__ import print_function
import os

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import backend as K
from snntoolbox.models.imagenet.utils import decode_predictions
from snntoolbox.models.imagenet.utils import preprocess_input

home_path = os.environ["HOME"]
data_path = os.path.join(home_path, ".snntoolbox")

TH_WEIGHTS_PATH = os.path.join(data_path,
                               'vgg19_weights_th_dim_ordering_th_kernels.h5')
TF_WEIGHTS_PATH = os.path.join(data_path,
                               'vgg19_weights_tf_dim_ordering_tf_kernels.h5')
TH_WEIGHTS_PATH_NO_TOP = os.path.join(
    data_path,
    'vgg19_weights_th_dim_ordering_th_kernels_notop.h5')
TF_WEIGHTS_PATH_NO_TOP = os.path.join(
    data_path,
    'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')


def VGG19(include_top=True, weights='imagenet',
          input_tensor=None):
    """Instantiate the VGG19 architecture.

    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_data_format() == 'channels_first':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, 3, 3, activation='relu', padding='same',
                      name='block1_conv1')(img_input)
    x = Conv2D(64, 3, 3, activation='relu', padding='same',
                      name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, 3, 3, activation='relu', padding='same',
                      name='block2_conv1')(x)
    x = Conv2D(128, 3, 3, activation='relu', padding='same',
                      name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, 3, 3, activation='relu', padding='same',
                      name='block3_conv1')(x)
    x = Conv2D(256, 3, 3, activation='relu', padding='same',
                      name='block3_conv2')(x)
    x = Conv2D(256, 3, 3, activation='relu', padding='same',
                      name='block3_conv3')(x)
    x = Conv2D(256, 3, 3, activation='relu', padding='same',
                      name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, 3, 3, activation='relu', padding='same',
                      name='block4_conv1')(x)
    x = Conv2D(512, 3, 3, activation='relu', padding='same',
                      name='block4_conv2')(x)
    x = Conv2D(512, 3, 3, activation='relu', padding='same',
                      name='block4_conv3')(x)
    x = Conv2D(512, 3, 3, activation='relu', padding='same',
                      name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, 3, 3, activation='relu', padding='same',
                      name='block5_conv1')(x)
    x = Conv2D(512, 3, 3, activation='relu', padding='same',
                      name='block5_conv2')(x)
    x = Conv2D(512, 3, 3, activation='relu', padding='same',
                      name='block5_conv3')(x)
    x = Conv2D(512, 3, 3, activation='relu', padding='same',
                      name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        print('k.image_data_format:', K.image_data_format())
        if K.image_data_format() == 'channels_first':
            if include_top:
                weights_path = TH_WEIGHTS_PATH
            else:
                weights_path = TH_WEIGHTS_PATH_NO_TOP
            model.load_weights(weights_path)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = TF_WEIGHTS_PATH
            else:
                weights_path = TF_WEIGHTS_PATH_NO_TOP
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    return model


if __name__ == '__main__':
    model = VGG19(include_top=True, weights='imagenet')

    img_path = './snntoolbox/models/imagenet/elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))

    filename = 'vgg19'
    open(filename + '.json', 'w').write(model.to_json())
    model.save_weights(filename + '.h5', overwrite=True)
