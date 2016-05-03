"""
Switching Simulators
--------------------

When running the SNN toolbox for the first time, it will create a configuration
file in your home directory:

``~/.snntoolbox/snntoolbox.json``

(You can of course create it yourself.)

It contains a dictionary of configuration options:

``{'simulator': 'brian'}``

Change the ``simulator`` key to any simulator you installed and which supports
pyNN. The modified settings will be loaded the next time you use any part of
the toolbox.

Simulators currently supported by pyNN include

    - 'nest'
    - 'brian'
    - 'Neuron'.

In addition, we provide our own simulator 'INI'.

"""

from __future__ import absolute_import, print_function
import os
import json
from importlib import import_module
from functools import partial

simulators_pyNN = {'nest', 'brian', 'Neuron'}
simulators_other = {'INI', 'brian2'}
simulators = simulators_pyNN.copy()
simulators.update(simulators_other)


def initialize_simulator(simulator):
    from snntoolbox.core.inisim import SpikeFlatten, SpikeDense
    from snntoolbox.core.inisim import SpikeConv2DReLU, AvgPool2DReLU
    custom_layers = {'SpikeFlatten': SpikeFlatten,
                     'SpikeDense': SpikeDense,
                     'SpikeConv2DReLU': SpikeConv2DReLU,
                     'AvgPool2DReLU': AvgPool2DReLU}
    if simulator in simulators_pyNN:
        sim = import_module('pyNN.' + simulator)
        from snntoolbox.config import cellparams, cellparams_pyNN
        from snntoolbox.config import simparams, simparams_pyNN
        cellparams.update(cellparams_pyNN)
        simparams.update(simparams_pyNN)
    elif simulator == 'brian2':
        sim = import_module('brian2')
        from snntoolbox.config import cellparams, cellparams_pyNN
        from snntoolbox.config import simparams, simparams_pyNN
        cellparams.update(cellparams_pyNN)
        simparams.update(simparams_pyNN)
    elif simulator == 'INI':
        sim = import_module('snntoolbox.core.inisim')
        print('\n')
        print("Heads-up: When using INI simulator, the batch size cannot be " +
              "changed after loading a previously converted spiking network " +
              "from file. To change the batch size, convert the ANN from " +
              "scratch.\n")
    print("Initialized {} simulator.\n".format(simulator))
    return sim, custom_layers


_base_dir = os.path.expanduser('~')
if not os.access(_base_dir, os.W_OK):
    _base_dir = '/tmp'

_dir = os.path.join(_base_dir, '.snntoolbox')
if not os.path.exists(_dir):
    os.makedirs(_dir)

_SIMULATOR = 'INI'
_config_path = os.path.expanduser(os.path.join(_dir, 'snntoolbox_config.json'))
if os.path.exists(_config_path):
    _config = json.load(open(_config_path))
    _sim = _config.get('simulator')
    assert _sim in simulators, \
        "Spiking neuron simulator '{}' ".format(_sim) + \
        "currently not supported by pyNN. Choose from {} ".format(simulators)
    _SIMULATOR = _sim
else:
    # Save config file, for easy edition
    _config = {'simulator': _SIMULATOR}
    with open(_config_path, 'w') as f:
        # Add new line in order for bash 'cat' display the content correctly
        f.write(json.dumps(_config) + '\n')

sim, custom_layers = initialize_simulator(_SIMULATOR)

# python 2 can not handle the 'flush' keyword argument of python 3 print().
# Provide 'echo' function as an alias for
# "print with flush and without newline".
try:
    echo = partial(print, end='', flush=True)
    echo(u'')
except TypeError:
    # TypeError: 'flush' is an invalid keyword argument for this function
    import sys

    def echo(text):
        """python 2 version of print(end='', flush=True)."""
        sys.stdout.write(u'{0}'.format(text))
        sys.stdout.flush()
