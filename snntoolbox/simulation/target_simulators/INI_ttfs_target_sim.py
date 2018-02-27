# -*- coding: utf-8 -*-
"""INI simulator with time-to-first-spike code.

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import sys
import keras
import numpy as np
from future import standard_library

from snntoolbox.simulation.target_simulators.INI_temporal_mean_rate_target_sim \
    import SNN as SNN_
standard_library.install_aliases()

remove_classifier = False


class SNN(SNN_):
    """
    The compiled spiking neural network, using layers derived from
    Keras base classes (see `snntoolbox.simulation.backends.inisim.ttfs`).

    Aims at simulating the network on a self-implemented Integrate-and-Fire
    simulator using a timestepped approach.
    """

    def __init__(self, config, queue=None):

        SNN_.__init__(self, config, queue)

    def simulate(self, **kwargs):

        from snntoolbox.utils.utils import echo
        from snntoolbox.simulation.utils import get_layer_synaptic_operations

        input_b_l = kwargs[str('x_b_l')] * self._dt

        output_b_l_t = np.zeros((self.batch_size, self.num_classes,
                                 self._num_timesteps))

        # Loop through simulation time.
        self.avg_rate = 0
        self._input_spikecount = 0
        for sim_step_int in range(self._num_timesteps):
            sim_step = (sim_step_int + 1) * self._dt
            self.set_time(sim_step)

            # Generate new input in case it changes with each simulation step.
            if self._poisson_input:
                input_b_l = self.get_poisson_frame_batch(kwargs[str('x_b_l')])
            elif self._dataset_format == 'aedat':
                input_b_l = kwargs[str('dvs_gen')].next_eventframe_batch()

            # Main step: Propagate input through network and record output
            # spikes.
            out_spikes = self.snn.predict_on_batch(input_b_l)

            # Add current spikes to previous spikes.
            if remove_classifier:  # Need to flatten output.
                output_b_l_t[:, :, sim_step_int] = np.argmax(np.reshape(
                    out_spikes > 0, (out_spikes.shape[0], -1)), 1)
            else:
                output_b_l_t[:, :, sim_step_int] = out_spikes > 0

            # Record neuron variables.
            i = j = 0
            for layer in self.snn.layers:
                # Excludes Input, Flatten, Concatenate, etc:
                if hasattr(layer, 'spiketrain') \
                        and layer.spiketrain is not None:
                    spiketrains_b_l = keras.backend.get_value(layer.spiketrain)
                    self.avg_rate += np.count_nonzero(spiketrains_b_l)
                    if self.spiketrains_n_b_l_t is not None:
                        self.spiketrains_n_b_l_t[i][0][
                            Ellipsis, sim_step_int] = spiketrains_b_l
                    if self.synaptic_operations_b_t is not None:
                        self.synaptic_operations_b_t[:, sim_step_int] += \
                            get_layer_synaptic_operations(spiketrains_b_l,
                                                          self.fanout[i + 1])
                    if self.neuron_operations_b_t is not None:
                        self.neuron_operations_b_t[:, sim_step_int] += \
                            self.num_neurons_with_bias[i + 1]
                    i += 1
                if hasattr(layer, 'mem') and self.mem_n_b_l_t is not None:
                    self.mem_n_b_l_t[j][0][Ellipsis, sim_step_int] = \
                        keras.backend.get_value(layer.mem)
                    j += 1

            if 'input_b_l_t' in self._log_keys:
                self.input_b_l_t[Ellipsis, sim_step_int] = input_b_l
            if self._poisson_input or self._dataset_format == 'aedat':
                if self.synaptic_operations_b_t is not None:
                    self.synaptic_operations_b_t[:, sim_step_int] += \
                        get_layer_synaptic_operations(input_b_l, self.fanout[0])
            else:
                if self.neuron_operations_b_t is not None:
                    if sim_step_int == 0:
                        self.neuron_operations_b_t[:, 0] += self.fanin[1] * \
                            self.num_neurons[1] * np.ones(self.batch_size) * 2

            spike_sums_b_l = np.sum(output_b_l_t, 2)
            undecided_b = np.sum(spike_sums_b_l, 1) == 0
            guesses_b = np.argmax(spike_sums_b_l, 1)
            none_class_b = -1 * np.ones(self.batch_size)
            clean_guesses_b = np.where(undecided_b, none_class_b, guesses_b)
            current_acc = np.mean(kwargs[str('truth_b')] == clean_guesses_b)
            if self.config.getint('output', 'verbose') > 0:
                if sim_step % 1 == 0:
                    echo('{:.2%}_'.format(current_acc))
            else:
                sys.stdout.write('\r{:>7.2%}'.format(current_acc))
                sys.stdout.flush()

            if self.config.getboolean('conversion', 'softmax_to_relu') and \
                    all(np.count_nonzero(output_b_l_t, (1, 2)) >= self.top_k):
                print("Finished early.")
                break

        self.avg_rate /= self.batch_size * np.sum(self.num_neurons) * \
            self._num_timesteps

        if self.spiketrains_n_b_l_t is None:
            print("Average spike rate: {} spikes per simulation time step."
                  "".format(self.avg_rate))

        return np.cumsum(output_b_l_t, 2)

    def load(self, path, filename):
        SNN_.load(self, path, filename)
