# -*- coding: utf-8 -*-
"""

Functions common to several input model parsers.

Created on Thu May 19 08:26:49 2016

@author: rbodo
"""


def border_mode_string(pad, pool_size):
    if pad == (0, 0):
        border_mode = 'valid'
    elif pad == (pool_size[0] // 2, pool_size[1] // 2):
        border_mode = 'same'
    elif pad == (pool_size[0] - 1, pool_size[1] - 1):
        border_mode = 'full'
    else:
        raise NotImplementedError(
            "Padding {} could not be interpreted as any of the ".format(pad) +
            "supported border modes 'valid', 'same' or 'full'.")
    return border_mode


def absorb_bn(w, b, gamma, beta, mean, std, epsilon):
    """
    Absorb the parameters of a batch-normalization layer into the previous
    layer.

    """

    import numpy as np
    ax = 0 if w.ndim < 3 else 1
    w_normed = w*np.swapaxes(gamma, 0, ax) / (np.swapaxes(std, 0, ax)+epsilon)
    b_normed = ((b - mean.flatten()) * gamma.flatten() /
                (std.flatten() + epsilon) + beta.flatten())
    return [w_normed, b_normed]


def import_script(path=None, filename=None):
    """
    Import python script which builds the model.

    Used if the input model library does not provide loading functions to
    restore a model from a saved file. In that case, use the script with which
    the model was compiled in the first place.

    Parameters
    ----------

        path: string, optional
            Path to directory where to load model from. Defaults to
            ``settings['path']``.

        filename: string, optional
            Name of file to load model from. Defaults to
            ``settings['filename']``.

    """

    import os
    import sys
    from snntoolbox.config import settings

    if path is None:
        path = settings['path']
    if filename is None:
        filename = settings['filename']

    v = sys.version_info
    if v >= (3, 5):
        import importlib.util
        filepath = os.path.join(path, filename + '.py')
        spec = importlib.util.spec_from_file_location(filename, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    elif v >= (3, 3):
        from importlib.machinery import SourceFileLoader
        mod = SourceFileLoader(filename, filepath).load_module()
    else:
        import imp
        mod = imp.load_source(filename, filepath)
    return mod
