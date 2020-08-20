# -*- coding: utf-8 -*-
"""
The main purpose of this module is to load a dataset from disk and feed it to
the toolbox in one of the formats it can handle.

For details see

.. autosummary::
    :nosignatures:

    get_dataset

@author: rbodo
"""

import json
import os

from configparser import NoOptionError
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from snntoolbox.utils.utils import import_helpers


def get_dataset(config):
    """Get dataset, either from ``.npz`` files or ``keras.ImageDataGenerator``.

    Returns Dictionaries with keys ``x_test`` and ``y_test`` if data set was
    loaded in ``.npz`` format, or with ``dataflow`` key if data will be loaded
    from ``.jpg``, ``.png``, or ``.bmp`` files by a
    ``keras.ImageDataGenerator``.

    Parameters
    ----------

    config: configparser.ConfigParser
        Settings.

    Returns
    -------

    normset: dict
        Used to normalized the network parameters.

    testset: dict
        Used to test the networks.

    """

    testset = {}
    normset = try_get_normset_from_scalefacs(config)
    dataset_path = config.get('paths', 'dataset_path')
    dataset_format = config.get('input', 'dataset_format')
    normalize_thresholds = config.getboolean('loihi', 'normalize_thresholds',
                                             fallback=False)
    is_testset_needed = config.getboolean('tools', 'evaluate_ann') or \
        config.getboolean('tools', 'simulate') or normalize_thresholds
    is_normset_needed = normalize_thresholds or (
            config.getboolean('tools', 'normalize') and normset == {})
    batch_size = config.getint('simulation', 'batch_size')

    # _______________________________ Keras __________________________________#
    try:
        keras_dataset = config.get('input', 'keras_dataset')
        if keras_dataset:
            from keras_rewiring.utilities.load_dataset \
                import load_and_preprocess_dataset
            num_to_test = config.getint('simulation', 'num_to_test')
            data = load_and_preprocess_dataset(keras_dataset)
            x_test, y_test = data['test']
            testset = {
                'x_test': x_test[:num_to_test],
                'y_test': y_test[:num_to_test]}
            if is_normset_needed:
                normset['x_norm'] = x_test
            return normset, testset
    except (NoOptionError, ImportError) as e:
        print("Warning:", e)

    # ________________________________ npz ___________________________________#
    if dataset_format == 'npz':
        print("Loading data set from '.npz' files in {}.\n".format(
            dataset_path))
        if is_testset_needed:
            num_to_test = config.getint('simulation', 'num_to_test')
            x_test = load_npz(dataset_path, 'x_test.npz')[:num_to_test]
            y_test = load_npz(dataset_path, 'y_test.npz')[:num_to_test]
            testset = {'x_test': x_test, 'y_test': y_test}
        if is_normset_needed:
            x_norm = load_npz(dataset_path, 'x_norm.npz')
            normset['x_norm'] = x_norm

    # ________________________________ jpg ___________________________________#
    elif dataset_format in {'jpg', 'png'}:
        print("Loading data set from ImageDataGenerator, using images in "
              "{}.\n".format(dataset_path))
        # Transform str to dict
        datagen_kwargs = eval(config.get('input', 'datagen_kwargs'))
        dataflow_kwargs = eval(config.get('input', 'dataflow_kwargs'))

        # Get class labels
        class_idx_path = config.get('paths', 'class_idx_path')
        if class_idx_path != '':
            class_idx = json.load(open(os.path.abspath(class_idx_path)))
            dataflow_kwargs['classes'] = \
                [class_idx[str(idx)][0] for idx in range(len(class_idx))]

        # Get proprocessing function
        if 'preprocessing_function' in datagen_kwargs:
            helpers = import_helpers(datagen_kwargs['preprocessing_function'],
                                     config)
            datagen_kwargs['preprocessing_function'] = \
                helpers.preprocessing_function

        dataflow_kwargs['directory'] = dataset_path
        if 'batch_size' not in dataflow_kwargs:
            dataflow_kwargs['batch_size'] = batch_size
        datagen = ImageDataGenerator(**datagen_kwargs)
        if (datagen.featurewise_center or datagen.featurewise_std_normalization
                or datagen.zca_whitening):
            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            rs = datagen_kwargs.get('rescale', None)
            x_orig = ImageDataGenerator(rescale=rs).flow_from_directory(
                **dataflow_kwargs).next()[0]
            datagen.fit(x_orig)
        if is_normset_needed:
            shuffle = dataflow_kwargs.get('shuffle')
            dataflow_kwargs['shuffle'] = True
            normset['dataflow'] = \
                datagen.flow_from_directory(**dataflow_kwargs)
            dataflow_kwargs['shuffle'] = shuffle
        if is_testset_needed:
            testset = {
                'dataflow': datagen.flow_from_directory(**dataflow_kwargs)}

    # _______________________________ aedat __________________________________#
    elif dataset_format == 'aedat':
        if is_normset_needed:
            print("Loading normalization dataset from '.npz' file in {}.\n"
                  "".format(dataset_path))
            x_norm = load_npz(dataset_path, 'x_norm.npz')
            normset['x_norm'] = x_norm
            # For Loihi threshold normalization we need to pass the
            # normalization data in the testset dict.
            testset = {'x_norm': x_norm}

    return normset, testset


def try_get_normset_from_scalefacs(config):
    """
    Instead of loading a normalization data set to calculate scale-factors, try
    to get the scale-factors stored on disk during a previous run.

    Parameters
    ----------

    config: configparser.ConfigParser
        Settings.

    Returns
    -------

    : Union[dict, None]
        A dictionary with single key 'scale_facs'. The corresponding value is
        itself a dictionary containing the scale factors for each layer.
        Returns empty set if no scale factors were found.
    """

    newpath = os.path.join(config.get('paths', 'log_dir_of_current_run'),
                           'normalization')
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    filepath = os.path.join(newpath, config.get('normalization',
                                                'percentile') + '.json')
    if os.path.isfile(filepath):
        print("Loading scale factors from disk instead of recalculating.")
        with open(filepath) as f:
            return {'scale_facs': json.load(f)}

    return {}


def to_categorical(y, nb_classes):
    """Convert class vector to binary class matrix.

    If the input ``y`` has shape (``nb_samples``,) and contains integers from 0
    to ``nb_classes``, the output array will be of dimension
    (``nb_samples``, ``nb_classes``).
    """

    y = np.asarray(y, dtype='int32')
    y_cat = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        y_cat[i, y[i]] = 1.
    return y_cat


def load_npz(path, filename):
    """Load dataset from an ``.npz`` file.

    Parameters
    ----------

    filename : string
        Name of file.
    path: string
        Location of dataset to load.

    Returns
    -------

    : tuple[np.array]
        The dataset as a numpy array containing samples.
    """

    return np.load(os.path.join(path, filename))['arr_0']
