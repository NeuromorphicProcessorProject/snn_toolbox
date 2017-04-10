# coding=utf-8

"""
Import common functions and initialize working directory.
"""

from __future__ import absolute_import, print_function

import os

# Set a base directory for the toolbox.
_base_dir = os.path.expanduser('~')
if not os.access(_base_dir, os.W_OK):
    _base_dir = '/tmp'

# Toolbox root directory.
toolbox_root = os.path.join(_base_dir, '.snntoolbox')
if not os.path.exists(toolbox_root):
    os.makedirs(toolbox_root)

# Path to toolbox preferences.
_config_path = os.path.join(toolbox_root, 'preferences')
if not os.path.exists(_config_path):
    os.makedirs(_config_path)

# Create a Keras config file so that even when importing Keras for the first
# time, the backend will be 'theano', not default 'tensorflow'.
_keras_base_dir = os.path.expanduser('~')
if not os.access(_keras_base_dir, os.W_OK):
    _keras_base_dir = '/tmp'
_keras_dir = os.path.join(_keras_base_dir, '.keras')
if not os.path.exists(_keras_dir):
    os.makedirs(_keras_dir)
_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if not os.path.exists(_config_path):
    import json
    _config = {'floatx': 'float32',
               'epsilon': 1e-07,
               'backend': 'theano',
               'image_data_format': 'channels_first'}
    with open(_config_path, 'w') as f:
        f.write(json.dumps(_config, indent=4))
