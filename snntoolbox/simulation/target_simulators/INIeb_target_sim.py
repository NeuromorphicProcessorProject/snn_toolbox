# coding=utf-8

"""Building SNNs using an event-driven simulator.

Created on Mon Jan 9 2017

@author: rbodo
"""

# For compatibility with python2
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
import sys
import warnings
from random import randint

import numpy as np
from future import standard_library
# noinspection PyUnresolvedReferences
from six.moves import cPickle
from snntoolbox.conversion.utils import echo
from snntoolbox.datasets.utils import confirm_overwrite
from typing import Optional

standard_library.install_aliases()


class SNN:
    """Class to hold the compiled spiking neural network.

    Class to hold the compiled spiking neural network, ready for testing in a
    spiking simulator.

    Attributes
    ----------

    sim: Simulator
        Module containing utility functions of spiking simulator. Result of
        calling ``snntoolbox.config.initialize_simulator()``. For instance, if
        using Brian simulator, this initialization would be equivalent to
        ``import pyNN.brian as sim``.

    layers: list
        Each entry represents a layer, i.e. a population of neurons, in form of
        pyNN ``Population`` objects.

    connections: list
        pyNN ``Projection`` objects representing the connections between
        individual layers.

    cellparams: dict
        Neuron cell parameters determining properties of the spiking neurons in
        pyNN simulators.

    Methods
    -------

    build:
        Convert an ANN to a spiking neural network, using layers derived from
        Keras base classes.
    run:
        Simulate a spiking network.
    save:
        Write model architecture and parameters to disk.
    load:
        Load model architecture and parameters from disk.
    end_sim:
        Clean up after simulation.
    """

    def __init__(self, s=None):
        """Init function."""

        if s is None:
            s = settings

        self.sim = initialize_simulator(s['simulator'])
        self.layers = []
        self.conns = []  # Temporary container for each layer.
        self.connections = []  # Final container for all layers.
        self.parsed_model = None

    # noinspection PyUnusedLocal
    def build(self, parsed_model, **kwargs):
        """
        Compile a spiking neural network to prepare for simulation.

        Written in pyNN (http://neuralensemble.org/docs/PyNN/).
        pyNN is a simulator-independent language for building neural network
        models. It allows running the converted net in a Spiking Simulator like
        Brian, NEURON, or NEST.

        During compilation, two lists are created and stored to disk:
        ``layers`` and ``connections``. Each entry in ``layers`` represents a
        population of neurons, given by a pyNN ``Population`` object. The
        neurons in these layers are connected by pyNN ``Projection`` s, stored
        in ``connections`` list.

        This compilation method performs the connection process between layers.
        This means, if the session was started with a call to ``sim.setup()``,
        the converted network can be tested right away, using the simulator
        ``sim``.

        However, when starting a new session (calling ``sim.setup()`` after
        conversion), the ``layers`` have to be reloaded from disk using
        ``load_assembly``, and the connections reestablished manually. This is
        implemented in ``run`` method, go there for details.
        See ``snntoolbox.core.pipeline.test_full`` about how to simulate after
        converting.

        Parameters
        ----------

        parsed_model: Keras model
            Parsed input model; result of applying
            ``model_lib.extract(input_model)`` to the ``input model``.
        """

        self.parsed_model = parsed_model

        print("Building spiking model...")

        self.add_input_layer(parsed_model.layers[0].batch_input_shape)

        # Iterate over layers to create spiking neurons and connections.
        for layer in parsed_model.layers[1:]:  # Skip input layer
            layer_type = layer.__class__.__name__
            if 'Flatten' in layer_type:
                continue
            print("Building layer: {}".format(layer.name))
            self.add_layer(layer)
            if layer_type == 'Dense':
                self.build_dense(layer)
            elif layer_type == 'Conv2D':
                self.build_convolution(layer)
            elif layer_type in {'MaxPooling2D', 'AveragePooling2D'}:
                self.build_pooling(layer)
            self.connect_layer()

        echo("Compilation finished.\n\n")

    def add_input_layer(self, input_shape):
        """Configure input layer."""

        self.layers.append(self.sim.Population(int(np.prod(input_shape[1:])),
                                               'InputLayer'))

    def add_layer(self, layer):
        """Add empty layer."""

        self.conns = []
        self.layers.append(self.sim.Population(
            int(np.prod(layer.output_shape[1:])), layer.name))

    def build_dense(self, layer):
        """Build dense layer."""

        [weights, biases] = layer.get_weights()
        for i in range(len(weights)):
            self.fanout_indices.append([j for j in range(len(weights[0]))])
            for j in range(len(weights[0])):
                self.conns.append((i, j, weights[i, j]))
