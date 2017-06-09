# -*- coding: utf-8 -*-
"""
Functions to load various properties of interest in analog and spiking neural
networks from disk.

Created on Wed Nov 18 13:38:46 2015

@author: rbodo
"""

# For compatibility with python2
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import json
import os

import h5py
import numpy as np
from future import standard_library

standard_library.install_aliases()


def get_dataset(config):
    """Get data set, either from ``.npz`` files or ``keras.ImageDataGenerator``.

    Returns Dictionaries with keys ``x_test`` and ``y_test`` if data set was
    loaded in ``.npz`` format, or with ``dataflow`` key if data will be loaded
    from ``.jpg`` files by a ``keras.ImageDataGenerator``.

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

    testset = None
    normset = try_get_normset_from_scalefacs(config)
    dataset_path = config['paths']['dataset_path']
    is_testset_needed = config.getboolean('tools', 'evaluate_ann') or \
        config.getboolean('tools', 'simulate')
    is_normset_needed = config.getboolean('tools', 'normalize') and \
        normset is None

    # ________________________________ npz ____________________________________#
    if config['input']['dataset_format'] == 'npz':
        print("Loading data set from '.npz' files in {}.\n".format(
            dataset_path))
        if is_testset_needed:
            testset = {'x_test': load_npz(dataset_path, 'x_test.npz'),
                       'y_test': load_npz(dataset_path, 'y_test.npz')}
            assert testset, "Test set empty."
        if is_normset_needed:
            normset = {'x_norm': load_npz(dataset_path, 'x_norm.npz')}
            assert normset, "Normalization set empty."

    # ________________________________ jpg ____________________________________#
    elif config['input']['dataset_format'] == 'jpg':
        from keras.preprocessing.image import ImageDataGenerator
        print("Loading data set from ImageDataGenerator, using images in "
              "{}.\n".format(dataset_path))
        # Transform str to dict
        datagen_kwargs = eval(config['input']['datagen_kwargs'])
        dataflow_kwargs = eval(config['input']['dataflow_kwargs'])

        # Get class labels
        class_idx_path = config['paths']['class_idx_path']
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
            dataflow_kwargs['batch_size'] = config.getint('simulation',
                                                          'batch_size')
        datagen = ImageDataGenerator(**datagen_kwargs)
        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        rs = datagen_kwargs['rescale'] if 'rescale' in datagen_kwargs else None
        x_orig = ImageDataGenerator(rescale=rs).flow_from_directory(
            **dataflow_kwargs).next()[0]
        datagen.fit(x_orig)
        if is_normset_needed:
            shuffle = dataflow_kwargs.get('shuffle')
            dataflow_kwargs['shuffle'] = True
            normset = {
                'dataflow': datagen.flow_from_directory(**dataflow_kwargs)}
            dataflow_kwargs['shuffle'] = shuffle
            assert normset, "Normalization set empty."
        if is_testset_needed:
            testset = {
                'dataflow': datagen.flow_from_directory(**dataflow_kwargs)}
            assert testset, "Test set empty."

    # _______________________________ aedat ___________________________________#
    elif config['input']['dataset_format'] == 'aedat':
        if is_normset_needed:
            normset = {'x_norm': load_npz(dataset_path, 'x_norm.npz')}
            assert normset, "Normalization set empty."
        testset = {}

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
        Returns ``None`` if no scale factors were found.
    """

    newpath = os.path.join(config['paths']['log_dir_of_current_run'],
                           'normalization')
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        return
    filepath = os.path.join(newpath, config['normalization']['percentile'] +
                            '.json')
    if os.path.isfile(filepath):
        print("Loading scale factors from disk instead of recalculating.")
        with open(filepath) as f:
            return {'scale_facs': json.load(f)}


def load_parameters(filepath):
    """Load all layer parameters from an HDF5 file."""

    f = h5py.File(filepath, 'r')

    params = []
    for k in sorted(f.keys()):
        params.append(np.array(f.get(k)))

    f.close()

    return params


def save_parameters(params, filepath, fileformat='h5'):
    """Save all layer parameters to an HDF5 file."""

    if fileformat == 'pkl':
        import pickle
        pickle.dump(params, open(filepath + '.pkl', str('wb')))
    else:
        with h5py.File(filepath, mode='w') as f:
            for i, p in enumerate(params):
                if i < 10:
                    j = '00' + str(i)
                elif i < 100:
                    j = '0' + str(i)
                else:
                    j = str(i)
                f.create_dataset('param_'+j, data=p)


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


def confirm_overwrite(filepath):
    """
    If config['output']['overwrite']==False and the file exists, ask user if it 
    should be overwritten.
    """

    if os.path.isfile(filepath):
        overwrite = input("[WARNING] {} already exists - ".format(filepath) +
                          "overwrite? [y/n]")
        while overwrite not in ['y', 'n']:
            overwrite = input("Enter 'y' (overwrite) or 'n' (cancel).")
        return overwrite == 'y'
    return True


def to_json(data, path):
    """Write ``data`` dictionary to ``path``.

    A ``TypeError`` is raised if objects in ``data`` are not JSON serializable.
    """

    def get_json_type(obj):
        """Get type of object to check if JSON serializable.

        Parameters
        ----------

        obj: object

        Raises
        ------

        TypeError

        Returns
        -------

        : Union(string, Any)
        """

        if type(obj).__module__ == np.__name__:
            # noinspection PyUnresolvedReferences
            return obj.item()

        # if obj is a python 'type'
        if type(obj).__name__ == type.__name__:
            return obj.__name__

        raise TypeError("{} not JSON serializable".format(type(obj).__name__))

    json.dump(data, open(path, str('w')), default=get_json_type)


def import_helpers(filepath, config):
    """Import a module with helper functions from ``filepath``.

    Parameters
    ----------

    filepath : str
        Filename or relative or absolute path of module to import. If only
        the filename is given, module is assumed to be in current working
        directory (``config['paths']['path_wd']). Non-absolute paths are taken
        relative to working dir.
    config : configparser.ConfigParser
        Settings.

    Returns
    -------

    :
        Module with helper functions.

    """

    from snntoolbox.model_libs.common import import_script

    path, filename = get_abs_path(filepath, config)

    return import_script(path, filename)


def get_abs_path(filepath, config):
    """Get an absolute path, possibly using current toolbox working dir.

    Parameters
    ----------

    filepath : str
        Filename or relative or absolute path. If only the filename is given,
        file is assumed to be in current working directory
        (``config['paths']['path_wd']). Non-absolute paths are interpreted
        relative to working dir.
    config : configparser.ConfigParser
        Settings.

    Returns
    -------

    path: str
        Absolute path to file.

    """

    path, filename = os.path.split(filepath)
    if path == '':
        path = config['paths']['path_wd']
    elif not os.path.isabs(path):
        path = os.path.abspath(os.path.join(config['paths']['path_wd'], path))
    return path, filename
