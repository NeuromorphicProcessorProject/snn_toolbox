# -*- coding: utf-8 -*-
"""INI simulator with temporal pattern code.

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import keras
import numpy as np
from future import standard_library

from snntoolbox.simulation.target_simulators.INI_temporal_mean_rate_target_sim \
    import SNN as SNN_
from snntoolbox.simulation.utils import get_layer_synaptic_operations

standard_library.install_aliases()

remove_classifier = False


class SNN(SNN_):
    """
    The compiled spiking neural network, using layers derived from
    Keras base classes (see `snntoolbox.simulation.backends.inisim.temporal_pattern`).

    Aims at simulating the network on a self-implemented Integrate-and-Fire
    simulator using a timestepped approach.

    Attributes
    ----------

    snn: keras.models.Model
        Keras model. This is the output format of the compiled spiking model
        because INI simulator runs networks of layers that are derived from
        Keras layer base classes.
    """

    def __init__(self, config, queue=None):

        SNN_.__init__(self, config, queue)

        self.num_bits = self.config.getint('conversion', 'num_bits')

    def compile(self):
        self.snn = keras.models.Model(
            self._input_images,
            self._spiking_layers[self.parsed_model.layers[-1].name])
        self.snn.compile('sgd', 'categorical_crossentropy', ['accuracy'])
        self.snn.set_weights(self.parsed_model.get_weights())
        for layer in self.snn.layers:
            if hasattr(layer, 'bias'):
                # Adjust biases to time resolution of simulator.
                bias = keras.backend.get_value(layer.bias) / self._num_timesteps
                keras.backend.set_value(layer.bias, bias)

    def simulate(self, **kwargs):

        from snntoolbox.utils.utils import echo

        input_b_l = kwargs[str('x_b_l')] * self._dt

        output_b_l_t = np.zeros((self.batch_size, self.num_classes,
                                 self._num_timesteps))

        self._input_spikecount = 0
        self.set_time(self._dt)

        # Main step: Propagate input through network and record output spikes.
        out_spikes = self.snn.predict_on_batch(input_b_l)

        # Add current spikes to previous spikes.
        x = self.sim.to_binary_numpy(out_spikes, self.num_bits)
        x *= np.expand_dims([2 ** -i for i in range(self.num_bits)], -1)
        output_b_l_t[:, :, :] = np.expand_dims(x.transpose(), 0)

        # Record neuron variables.
        i = 0
        for layer in self.snn.layers:
            # Excludes Input, Flatten, Concatenate, etc:
            if hasattr(layer, 'spikerates') and layer.spikerates is not None:
                spikerates_b_l = keras.backend.get_value(layer.spikerates)
                spiketrains_b_l_t = self.spikerates_to_trains(spikerates_b_l)
                self.set_spikerates(spikerates_b_l, i)
                self.set_spiketrains(spiketrains_b_l_t, i)
                if self.synaptic_operations_b_t is not None:
                    self.set_synaptic_operations(spiketrains_b_l_t, i)
                if self.neuron_operations_b_t is not None:
                    self.set_neuron_operations(i)
                i += 1
        if 'input_b_l_t' in self._log_keys:
            self.input_b_l_t[Ellipsis, 0] = input_b_l
        if self.neuron_operations_b_t is not None:
            self.neuron_operations_b_t[:, 0] += self.fanin[1] * \
                self.num_neurons[1] * np.ones(self.batch_size) * 2

        if self.config.getint('output', 'verbose') > 0:
            guesses_b = np.argmax(np.sum(output_b_l_t, 2), 1)
            echo('{:.2%}_'.format(np.mean(kwargs[str('truth_b')] == guesses_b)))

        return np.cumsum(output_b_l_t, 2)

    def load(self, path, filename):
        SNN_.load(self, path, filename)

    def set_spiketrains(self, spiketrains_b_l_t, i):
        if self.spiketrains_n_b_l_t is not None:
            self.spiketrains_n_b_l_t[i][0][:] = spiketrains_b_l_t

    def set_spikerates(self, spikerates_b_l, i):
        if self.spikerates_n_b_l is not None:
            self.spikerates_n_b_l[i][0][:] = spikerates_b_l

    def set_neuron_operations(self, i):
        self.neuron_operations_b_t += self.num_neurons_with_bias[i + 1]

    def set_synaptic_operations(self, spiketrains_b_l_t, i):
        for t in range(self.synaptic_operations_b_t.shape[-1]):
            self.synaptic_operations_b_t[:, t] += 2 * \
                get_layer_synaptic_operations(
                    spiketrains_b_l_t[Ellipsis, t], self.fanout[i + 1])

    def spikerates_to_trains(self, spikerates_b_l):
        x = self.sim.to_binary_numpy(spikerates_b_l, self.num_bits)
        shape = [self.num_bits] + [1] * (x.ndim - 1)
        x *= np.resize(np.arange(self.num_bits), shape)
        perm = (1, 2, 3, 0) if len(x.shape) > 2 else (1, 0)
        spiketrains_b_l_t = np.expand_dims(np.transpose(x, perm), 0)
        return spiketrains_b_l_t
