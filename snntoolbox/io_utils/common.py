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
from typing import Optional

standard_library.install_aliases()


def load_parameters(filepath):
    """Load all layer parameters from an HDF5 file."""

    f = h5py.File(filepath, mode='r')

    params = []
    for k in f.keys():
        params.append(np.array(f.get(k)))

    f.close()

    return params


def save_parameters(params, filepath):
    """Save all layer parameters to an HDF5 file."""

    with h5py.File(filepath, mode='w') as f:
        for i, p in enumerate(params):
            j = '0' + str(i) if i < 10 else str(i)
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
    """Load dataset from an ``.npy`` or ``.npz`` file.

    Parameters
    ----------

    filename : string
        Name of file.
    path: string
        Location of dataset to load.

    Returns
    -------

    : tuple[np.array]
        The dataset as a numpy array containing samples. Example:
        With original data of the form (channels, num_rows, num_cols),
        ``x_train`` and ``x_test`` have dimension
        (num_samples, channels*num_rows*num_cols) for a fully-connected network,
        and (num_samples, channels, num_rows, num_cols) otherwise.
        ``y_train`` and ``y_test`` have dimension (num_samples, num_classes).
    """

    return np.load(os.path.join(path, filename))['arr_0']


def download_dataset(fname, origin, untar=False):
    """Download a dataset, if not already there.

    Parameters
    ----------

    fname: str
        Full filename of dataset, e.g. ``mnist.pkl.gz``.
    origin: str
        Location of dataset, e.g. url
        https://s3.amazonaws.com/img-datasets/mnist.pkl.gz
    untar: Optional[bool]
        If ``True``, untar file.

    Returns
    -------

    fpath: str
        The path to the downloaded dataset. If the user has write access to
        ``home``, the dataset will be stored in ``~/.snntoolbox/datasets/``,
        otherwise in ``/tmp/.snntoolbox/datasets/``.

    Notes
    -----

    Test under python2.
    """

    import tarfile
    import shutil
    from six.moves.urllib.error import URLError, HTTPError
    # Under Python 2, 'urlretrieve' relies on FancyURLopener from legacy
    # urllib module, known to have issues with proxy management
    from six.moves.urllib.request import urlretrieve

    datadir_base = os.path.expanduser(os.path.join('~', '.snntoolbox'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.snntoolbox')
    datadir = os.path.join(datadir_base, 'datasets')
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    untar_fpath = None
    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        print("Downloading data from {}".format(origin))
        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise e

    if untar:
        if not os.path.exists(untar_fpath):
            print("Untaring file...\n")
            tfile = tarfile.open(fpath, 'r:gz')
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise e
            tfile.close()
        return untar_fpath

    return fpath


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
