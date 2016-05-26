# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:26:49 2016

@author: rbodo
"""


def absorb_bn(w, b, gamma, beta, mean, std, epsilon):
    import numpy as np
    ax = 0 if w.ndim < 3 else 1
    w_normed = w*np.swapaxes(gamma, 0, ax) / (np.swapaxes(std, 0, ax)+epsilon)
    b_normed = ((b - mean.flatten()) * gamma.flatten() /
                (std.flatten() + epsilon) + beta.flatten())
    return [w_normed, b_normed]


def import_script(filename):
    # Import script which builds the model
    import os
    import sys
    from snntoolbox.config import settings
    v = sys.version_info
    if v >= (3, 5):
        import importlib.util
        path = os.path.join(settings['path'], filename + '.py')
        spec = importlib.util.spec_from_file_location(filename, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    elif v >= (3, 3):
        from importlib.machinery import SourceFileLoader
        mod = SourceFileLoader(filename, path).load_module()
    else:
        import imp
        mod = imp.load_source(filename, path)
    return mod
