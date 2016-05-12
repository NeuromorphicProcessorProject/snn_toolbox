# -*- coding: utf-8 -*-
"""
``convert_to_SNN`` converts an `analog` to a `spiking` neural network.

Returns the spiking model in the simulator-independent language pyNN.

Created on Thu Dec 10 09:29:29 2015

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import numpy as np

from snntoolbox import sim, echo
from snntoolbox.config import globalparams, cellparams, simparams

standard_library.install_aliases()


def convert_to_SNN(ann):
    import snntoolbox
    s = snntoolbox._SIMULATOR
    if s == 'brian2' or s in snntoolbox.simulators_pyNN:
        return convert_to_SNN_pyNN(ann)
    elif s == 'INI':
        return convert_to_SNN_keras(ann)


def convert_to_SNN_pyNN(ann):
    """
    Convert an `analog` to a `spiking` neural network.

    Written in pyNN (http://neuralensemble.org/docs/PyNN/).
    pyNN is a simulator-independent language for building neural network
    models. It allows running the converted net in a Spiking Simulator like
    Brian, NEURON, or NEST.

    During conversion, two lists are created and stored to disk: ``layers`` and
    ``connections``. Each entry in ``layers`` represents a population of
    neurons, given by a pyNN ``Population`` object. The neurons in these layers
    are connected by pyNN ``Projection`` s, stored in ``connections`` list.
    This conversion method performs the connection process between layers. This
    means, if the session was started with a call to ``sim.setup()``, the
    converted network can be tested right away, using the simulator ``sim``.
    However, when starting a new session (calling ``sim.setup()`` after
    conversion), the ``layers`` have to be reloaded from disk using
    ``io.load.load_assembly``, and the connections reestablished
    manually. This is implemented in ``simulation.run_SNN``, go there for
    details. See ``tests.util.test_full`` about how to simulate after
    converting. The script ``tests.parameter_sweep`` wraps all these functions
    in one and allows toggling them by setting a few parameters.

    Parameters
    ----------
    ann : ANN model
        The network architecture and weights as ``snntoolbox.io.load.ANN``
        object.

    Returns
    -------
    layers : list
        Each entry represents a layer, i.e. a population of neurons, in form of
        pyNN ``Population`` objects.
    """

    import snntoolbox
    from snntoolbox.io.save import save_assembly, save_connections

    if not snntoolbox._SIMULATOR == 'brian2':
        # Setup simulator, in case the user wants to test the network right
        # after converting.
        # From the pyNN documentation:
        # "Before using any other functions or classes from PyNN, the user must
        # call the setup() function. Calling setup() a second time resets the
        # simulator entirely, destroying any network that may have been created
        # in the meantime."
        sim.setup(timestep=simparams['dt'])

    echo('\n')
    echo("Compiling spiking network...\n")

    # Create new container for the converted layers and insert the input layer.
    if snntoolbox._SIMULATOR == 'brian2':
        layers = [sim.PoissonGroup(np.prod(ann.input_shape[1:]),
                                   rates=0*sim.Hz,
                                   dt=simparams['dt']*sim.ms)]
        # Define differential equation of membrane potential
        threshold = 'v > v_thresh'
        reset = 'v = v_reset'
        eqs = 'dv/dt = -v/tau_m : volt'
        spikemonitors = [sim.SpikeMonitor(layers[0])]
        statemonitors = []
        labels = [ann.labels[0]]
    else:
        layers = [sim.Population(int(np.prod(ann.input_shape[1:])),
                                 sim.SpikeSourcePoisson(),
                  label=ann.labels[0])]
    # Create new container for the connections between layers.
    connections = []
    # Iterate over hidden layers to create spiking neurons and store
    # connections.
    for layer in ann.layers:
        layer_type = layer['layer_type']
        if layer_type in {'Dense', 'Convolution2D', 'MaxPooling2D',
                          'AveragePooling2D'}:
            conns = []
            echo("Building layer: {}\n".format(layer['label']))
            if snntoolbox._SIMULATOR == 'brian2':
                labels.append(layer['label'])
                layers.append(sim.NeuronGroup(
                    np.prod(layer['output_shape'][1:]), model=eqs,
                    threshold=threshold, reset=reset,
                    dt=simparams['dt']*sim.ms, method='linear'))
                connections.append(sim.Synapses(layers[-2], layers[-1],
                                   model='w:volt', on_pre='v+=w',
                                   dt=simparams['dt']*sim.ms))
                if globalparams['verbose'] > 1:
                    spikemonitors.append(sim.SpikeMonitor(layers[-1]))
                if globalparams['verbose'] == 3:
                    statemonitors.append(sim.StateMonitor(layers[-1], 'v',
                                                          record=True))
            else:
                layers.append(sim.Population(
                              int(np.prod(layer['output_shape'][1:])),
                              sim.IF_cond_exp, cellparams,
                              label=layer['label']))
        else:
            echo("Skipped layer:  {}\n".format(layer_type))
            continue
        if layer_type in {'Dense', 'Convolution2D'}:
            weights = layer['weights'][0]  # [W, b][0]
            if layer_type == 'Dense':
                if snntoolbox._SIMULATOR == 'brian2':
                    connections[-1].connect(True)
                    connections[-1].w = weights.flatten() * sim.volt
                else:
                    for i in range(layers[-2].size):
                        for j in range(layers[-1].size):
                            conns.append((i, j, weights[i, j],
                                          simparams['delay']))
            elif layer_type == 'Convolution2D':
                nx = layer['input_shape'][3]  # Width of feature map
                ny = layer['input_shape'][2]  # Hight of feature map
                kx = layer['nb_col']  # Width of kernel
                ky = layer['nb_row']  # Hight of kernel
                px = int((kx - 1) / 2)  # Zero-padding columns
                py = int((ky - 1) / 2)  # Zero-padding rows
                if layer['border_mode'] == 'valid':
                    # In border_mode 'valid', the original sidelength is
                    # reduced by one less than the kernel size.
                    mx = nx - kx + 1  # Number of columns in output filters
                    my = ny - ky + 1  # Number of rows in output filters
                    x0 = px
                    y0 = py
                elif layer['border_mode'] == 'same':
                    mx = nx
                    my = ny
                    x0 = 0
                    y0 = 0
                else:
                    raise Exception("Border_mode {} not supported".
                                    format(layer['border_mode']))
                # Loop over output filters 'fout'
                for fout in range(weights.shape[0]):
                    for y in range(y0, ny - y0):
                        for x in range(x0, nx - x0):
                            target = x - x0 + (y - y0) * mx + fout * mx * my
                            # Loop over input filters 'fin'
                            for fin in range(weights.shape[1]):
                                for k in range(-py, py + 1):
                                    if not 0 <= y + k < ny:
                                        continue
                                    source = x + (y + k) * nx + fin * nx * ny
                                    for l in range(-px, px + 1):
                                        if not 0 <= x + l < nx:
                                            continue
                                        if snntoolbox._SIMULATOR == 'brian2':
                                            connections[-1].connect(i=source+l,
                                                                    j=target)
                                            connections[-1].w[source + l,
                                                              target] = \
                                                weights[fout, fin,
                                                        py-k, px-l] * sim.volt
                                        else:
                                            conns.append((source + l, target,
                                                          weights[fout, fin,
                                                                  py-k, px-l],
                                                          simparams['delay']))
                        echo('.')
                    echo(' {:.1%}\n'.format(((fout + 1) * weights.shape[1]) /
                         (weights.shape[0] * weights.shape[1])))
        elif layer_type in {'MaxPooling2D', 'AveragePooling2D'}:
            if layer_type == 'MaxPooling2D':
                echo("WARNING: Layer type 'MaxPooling' not supported yet. \
                     Falling back on 'AveragePooling'.\n")
            nx = layer['input_shape'][3]  # Width of feature map
            ny = layer['input_shape'][2]  # Hight of feature map
            dx = layer['pool_size'][1]  # Width of pool
            dy = layer['pool_size'][0]  # Hight of pool
            sx = layer['strides'][1]
            sy = layer['strides'][0]
            for fout in range(layer['input_shape'][1]):  # Feature maps
                for y in range(0, ny - dy + 1, sy):
                    for x in range(0, nx - dx + 1, sx):
                        target = int(x / sx + y / sy * ((nx - dx) / sx + 1) +
                                     fout * nx * ny / (dx * dy))
                        for k in range(dy):
                            source = x + (y + k) * nx + fout * nx * ny
                            for l in range(dx):
                                if snntoolbox._SIMULATOR == 'brian2':
                                    connections[-1].connect(i=source+l,
                                                            j=target)
                                else:
                                    conns.append((source + l, target,
                                                  1 / (dx * dy),
                                                 simparams['delay']))
                    echo('.')
                echo(' {:.1%}\n'.format((1 + fout) / layer['input_shape'][1]))
            if snntoolbox._SIMULATOR == 'brian2':
                connections[-1].w = sim.volt / (dx * dy)
        if not snntoolbox._SIMULATOR == 'brian2':
            # Turn off warning because we have no influence on it:
            # "UserWarning: ConvergentConnect is deprecated and will be removed
            # in a future version of NEST. Please use Connect instead!"
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                warnings.warn('deprecated', UserWarning)
                # Connect layers
                connections.append(sim.Projection(layers[-2], layers[-1],
                                                  sim.FromListConnector(
                                                  conns, ['weight', 'delay'])))

    echo("Compilation finished.\n\n")

    if snntoolbox._SIMULATOR == 'brian2':
        # Track the output layer spikes. Add monitor here if it was not already
        # appended above (because globalparams['verbose'] < 1)
        if len(spikemonitors) < len(layers):
            spikemonitors.append(sim.SpikeMonitor(layers[-1]))
        # Create snapshot of network
        snn = sim.Network(layers, connections, spikemonitors, statemonitors)
        return {'snn_brian2': snn, 'labels': labels,
                'spikemonitors': spikemonitors, 'statemonitors': statemonitors}

    save_assembly(layers)
    save_connections(connections)
    return {'snn_pyNN': layers}


def convert_to_SNN_keras(ann):
    """
    Convert an ANN to a spiking neural network, using layers derived from
    Keras base classes.

    Aims at simulating the network on a self-implemented Integrate-and-Fire
    simulator using mean pooling and a timestepped approach.

    Parameters
    ----------
    ann : Keras model
        The network architecture and weights in json format.

    Returns
    -------
    snn : Keras model (derived class of Sequential)
        The spiking model
    get_output : Theano function
        Computes the output of the network
    """

    import theano
    import theano.tensor as T
    from keras.models import Sequential
    from snntoolbox.io.save import save_model

    # Create empty spiking network
    snn = Sequential()
    # Get dimensions of the input
    input_shape = list(ann.input_shape)
    input_shape[0] = globalparams['batch_size']

    # Allocate input variables
    input_time = T.scalar('time')

    # Iterate over layers to create spiking neurons and connections.
    if globalparams['verbose'] > 1:
        echo("Iterating over ANN layers to add spiking layers...\n")
    for (layer_num, layer) in enumerate(ann.layers):
        layer_type = layer['layer_type']
        kwargs = {'name': layer['label'], 'trainable': False}
        kwargs2 = {}
        if layer_num == 0:
            # For the input layer, pass extra keyword argument
            # 'batch_input_shape' to layer constructor.
            kwargs.update({'batch_input_shape': input_shape})
            kwargs2.update({'time_var': input_time})
        echo("Layer: {}\n".format(layer['label']))
        if layer_type == 'Convolution2D':
            snn.add(sim.SpikeConv2DReLU(layer['nb_filter'],
                                        layer['nb_row'], layer['nb_col'],
                                        sim.floatX(layer['weights']),
                                        border_mode=layer['border_mode'],
                                        **kwargs))
        elif layer_type == 'Dense':
            snn.add(sim.SpikeDense(layer['output_shape'][1],
                                   sim.floatX(layer['weights']), **kwargs))
        elif layer_type == 'Flatten':
            snn.add(sim.SpikeFlatten(**kwargs))
        elif layer_type in {'MaxPooling2D', 'AveragePooling2D'}:
            snn.add(sim.AvgPool2DReLU(pool_size=layer['pool_size'],
                                      strides=layer['strides'],
                                      border_mode=layer['border_mode'],
                                      label=layer['label']))
        if layer_type in {'Convolution2D', 'Dense', 'Flatten', 'MaxPooling2D',
                          'AveragePooling2D'}:
            sim.init_neurons(snn.layers[-1], v_thresh=cellparams['v_thresh'],
                             tau_refrac=cellparams['tau_refrac'], **kwargs2)

    # Compile
    echo('\n')
    echo("Compiling spiking network...\n")
    snn.compile(loss='categorical_crossentropy', optimizer='sgd',
                metrics=['accuracy'])
    output_spikes = snn.layers[-1].get_output()
    output_time = sim.get_time(snn.layers[-1])
    updates = sim.get_updates(snn.layers[-1])
    get_output = theano.function([snn.input, input_time],
                                 [output_spikes, output_time], updates=updates)
    echo("Compilation finished.\n\n")
    save_model(snn, filename='snn_keras_' + globalparams['filename'],
               spiking=True)

    return {'snn_keras': snn, 'get_output': get_output}
