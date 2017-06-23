# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:58:15 2015

@author: rbodo
"""


def createPoissonSpikeInput(X_test, ind, layers):
    """
    Function taken from ``simulation.run_SNN()``. Replaced ``SpikeSourceArray``
    with ``SpikeSourcePoisson``, which is faster and easier to debug.
    """

    import numpy as np
    from pyNN.parameters import Sequence
    from snntoolbox.utils.config import simparams

    dt = int(np.ceil(simparams['dt']))
    duration = int(simparams['duration'])
    spike_list = []
    # Loop over simulation time with temporal resolution dt to determine
    # when the test sample causes spikes in the input layer.
    # Sidenote: The Nest simulator has several restrictions on the
    # spike times: see http://www.nest-simulator.org/cc/spike_generator/.
    # For these reasons we shift the spike times about 2dt.
    for t in range(2 * dt, duration + 2 * dt, dt):
        # Draw a random sample of the same size as the input sample.
        spike_snapshot = \
            np.random.random_sample(int(np.prod(X_test[0, :].shape)))
        # Fire a spike at time dt if entry in input sample (flattened to 1D
        # and multiplied by maximum firing rate) exceeds random number.
        # (Array of booleans)
        spikes = spike_snapshot <= X_test[ind, :].flatten()
        # Convert array of booleans to array of floats indicating the
        # precise timing of the spikes. Append to the list of spike times,
        # where each row corresponds to a further step in the simulation.
        spike_list.append(spikes * (t + 0.001))
    # Here spike_list becomes a 2D array of shape [input_size, duration/dt]
    spike_array = np.array(spike_list).transpose()
    # To be able to feed it to the input layer, convert it to pyNN
    # Sequence type. The number of entries in the new container
    # spike_sequences equals the size of the input sample, and each entry
    # is a Sequence of nonzero, increasing spike times.
    spike_sequences = []
    for i in range(len(spike_array)):
        spike_sequences.append(Sequence([j for j in spike_array[i, :]
                                         if j != 0]))
    # Insert poisson input.
    layers[0].set(spike_times=spike_sequences)


# This used to be part of the __init__.py file in the root directory of
# snntoolbox.

def switch_simulator(_config_path):
    """
    Switching Simulators
    --------------------

    When running the SNN toolbox for the first time, it will create a
    configuration file in your home directory:

    ``~/.snntoolbox/snntoolbox.json``

    (You can of course create it yourself.)

    It contains a dictionary of configuration options:

    ``{'simulator': 'brian'}``

    Change the ``simulator`` key to any simulator you installed and which
    supports pyNN. The modified settings will be loaded the next time you use
    any part of the toolbox.

    Simulators currently supported by pyNN include

        - 'nest'
        - 'brian'
        - 'Neuron'.

    In addition, we provide our own simulator 'INI'.

    """

    import os
    import json

    _config_file = os.path.join(_config_path, 'snntoolbox_config.json')
    if os.path.exists(_config_file):
        _config = json.load(open(_config_file))
        _sim = _config.get('simulator')
        _SIMULATOR = _sim
    else:
        # Save config file, for easy edition
        _config = {'simulator': _SIMULATOR}
        with open(_config_file, 'w') as f:
            # Add new line in order for bash 'cat' display the content
            # correctly
            f.write(json.dumps(_config) + '\n')


def plot_hot_neurons(self, activations, idx):
    import numpy as np
    import matplotlib.pyplot as plt
    from snntoolbox.utils.config import settings

    plt.figure()
    plt.hist(np.max(
        activations, axis=tuple(range(1, activations.ndim))),
        bins=len(activations))
    plt.title("Distribution of maximum activations \n in layer " +
              "{}".format(self.labels[idx-1]))
    plt.xlabel("max activation")
    plt.ylabel("sample count")
    plt.show()
    if 'Dense' in self.labels[idx-1]:
        plt.figure()
        p = np.percentile(activations, settings['percentile'])
        plt.hist(np.nonzero(activations >= p)[1],
                 bins=activations.shape[1])
        plt.title("Histogram of 'hot' neurons\n" +
                  "in  layer {}".format(self.labels[idx-1]))
        plt.text(0.8, 0.8, "percentile = {}\nscale = {:.2f}".format(
            settings['percentile'], p))
        plt.xlabel("Neuron index")
        plt.ylabel("Sample count")
        plt.show()
