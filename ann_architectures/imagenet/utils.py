# coding=utf-8

"""Utility functions for ImageNet testing."""

import numpy as np
import json

from keras.utils.data_utils import get_file
from keras import backend as k

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/' + \
    'image-models/imagenet_class_index.json'


def preprocess_input(x, image_data_format='default'):
    """Preprocess input.

    Parameters
    ----------
    x: np.array
        The input samples.
    image_data_format: str
        Ordering of the dimensions. ``'th'`` for Theano, ``'tf'`` for
        TensorFlow.

    Returns
    -------

    x: np.array
        The processed data.
    """

    if image_data_format == 'default':
        image_data_format = k.image_data_format()
    assert image_data_format in {'channels_first', 'channels_last'}

    if image_data_format == 'channels_first':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def decode_predictions(preds):
    """Decode predictions.

    Parameters
    ----------
    preds: np.array
        The predictions.

    Returns
    -------

    results: list
        The class indices.
    """

    global CLASS_INDEX
    assert len(preds.shape) == 2 and preds.shape[1] == 1000
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    indices = np.argmax(preds, axis=-1)
    results = []
    for i in indices:
        results.append(CLASS_INDEX[str(i)])
    return results
