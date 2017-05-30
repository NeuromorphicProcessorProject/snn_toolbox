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
from snntoolbox.config import settings

standard_library.install_aliases()


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
        pickle.dump(params, open(filepath + '.pkl', 'wb'))
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


def load_dataset(path, filename):
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
    If settings['overwrite']==False and the file exists, ask user if it should
    be overwritten.
    """

    if not settings['overwrite'] and os.path.isfile(filepath):
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

    json.dump(data, open(path, 'w'), default=get_json_type)
