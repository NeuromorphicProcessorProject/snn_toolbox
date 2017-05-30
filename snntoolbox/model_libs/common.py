# -*- coding: utf-8 -*-
"""

Functions common to several input model parsers.

Created on Thu May 19 08:26:49 2016

@author: rbodo
"""

import numpy as np


def padding_string(pad, pool_size):
    """Get string defining the border mode.

    Parameters
    ----------
    pad: tuple[int]
        Zero-padding in x- and y-direction.
    pool_size: list[int]
        Size of kernel.

    Returns
    -------

    padding: str
        Border mode identifier.
    """

    if pad == (0, 0):
        padding = 'valid'
    elif pad == (pool_size[0] // 2, pool_size[1] // 2):
        padding = 'same'
    elif pad == (pool_size[0] - 1, pool_size[1] - 1):
        padding = 'full'
    else:
        raise NotImplementedError(
            "Padding {} could not be interpreted as any of the ".format(pad) +
            "supported border modes 'valid', 'same' or 'full'.")
    return padding


def absorb_bn(w, b, gamma, beta, mean, var, epsilon, axis=None):
    """
    Absorb the parameters of a batch-normalization layer into the previous
    layer.
    """

    if axis is None:
        axis = 0 if w.ndim > 2 else 1

    broadcast_shape = [1] * w.ndim  # e.g. [1, 1, 1, 1] for ConvLayer
    broadcast_shape[axis] = w.shape[axis]  # [64, 1, 1, 1] for 64 features
    var_broadcast = np.reshape(var, broadcast_shape)
    gamma_broadcast = np.reshape(gamma, broadcast_shape)

    b_bn = beta + (b - mean) * gamma / np.sqrt(var + epsilon)
    w_bn = w * gamma_broadcast / np.sqrt(var_broadcast + epsilon)

    return w_bn, b_bn


def import_script(path, filename):
    """Import python script independently from python version.

    Parameters
    ----------

    path: string
        Path to directory where to load script from.

    filename: string
        Name of script file.
    """

    import os
    import sys

    filepath = os.path.join(path, filename + '.py')

    v = sys.version_info
    if v >= (3, 5):
        import importlib.util
        spec = importlib.util.spec_from_file_location(filename, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    elif v >= (3, 3):
        # noinspection PyCompatibility
        from importlib.machinery import SourceFileLoader
        mod = SourceFileLoader(filename, filepath).load_module()
    else:
        # noinspection PyDeprecation
        import imp
        # noinspection PyDeprecation
        mod = imp.load_source(filename, filepath)
    return mod
