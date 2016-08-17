# -*- coding: utf-8 -*-
"""
Functions to save various properties of interest in analog and spiking neural
networks to disk.

Created on Wed Nov 18 11:57:29 2015

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import os
import numpy as np
import json
from snntoolbox.config import settings

standard_library.install_aliases()


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
    """
    Write ``data`` dictionary to ``path``.

    A ``TypeError`` is raised if objects in ``data`` are not JSON serializable.

    """

    def get_json_type(obj):
        # if obj is any numpy type
        if type(obj).__module__ == np.__name__:
            return obj.item()

        # if obj is a python 'type'
        if type(obj).__name__ == type.__name__:
            return obj.__name__

        raise TypeError("{} not JSON serializable".format(type(obj).__name__))

    json.dump(data, open(path, 'w'), default=get_json_type)
