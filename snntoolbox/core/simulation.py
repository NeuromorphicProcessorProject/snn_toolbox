# -*- coding: utf-8 -*-
"""
``run_SNN`` simulates a spiking network written in pyNN using a specified
simulator (pyNN currently supports Brian, NEST, NEURON).

Created on Fri Dec 18 16:58:15 2015

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library
from builtins import int, range

import os
import warnings
import numpy as np
from random import randint
import snntoolbox
from snntoolbox import sim, echo
from snntoolbox.config import globalparams, cellparams, simparams
from snntoolbox.io.plotting import plot_spiketrains, plot_layer_activity
from snntoolbox.io.plotting import plot_layer_summaries
from snntoolbox.io.plotting import plot_pearson_coefficients
from snntoolbox.io.load import load_assembly, load_model
from snntoolbox.core.util import extract_label

standard_library.install_aliases()


def run_SNN(X_test, Y_test, **kwargs):
    s = snntoolbox._SIMULATOR
    if s == 'brian2' or s in snntoolbox.simulators_pyNN:
        return run_SNN_pyNN(X_test, Y_test, **kwargs)
    elif snntoolbox._SIMULATOR == 'INI':
        return run_SNN_keras(X_test, Y_test, **kwargs)


def run_SNN_pyNN(X_test, Y_test, **kwargs):
    """
    Simulate a spiking network with IF units and Poisson input in pyNN, using
    a simulator like Brian, NEST, NEURON, etc.

    This function will randomly select ``simparams['num_to_test']`` test
    samples among ``X_test`` and simulate the network on those.

    If ``globalparams['verbose'] > 1``, the simulator records the spiketrains
    and membrane potential of each neuron in each layer. Doing so for all
    ``simparams['num_to_test']`` test samples is very costly in terms of memory
    and time, but can be useful for debugging the network's general
    functioning. This function returns only the recordings of the last test
    sample. To get detailed information about the network's behavior for a
    particular sample, replace the test set ``X_test, Y_test`` with this sample
    of interest.

    Parameters
    ----------

    X_test : float32 array
        The input samples to test.
        With data of the form (channels, num_rows, num_cols),
        X_test has dimension (num_samples, channels*num_rows*num_cols)
        for a multi-layer perceptron, and
        (num_samples, channels, num_rows, num_cols) for a convolutional net.
    Y_test : float32 array
        Ground truth of test data. Has dimension (num_samples, num_classes).
    layers : list, possible kwarg
        Each entry represents a layer, i.e. a population of neurons, in form of
        pyNN ``Population`` objects.

    Returns
    -------

    total_acc : float
        Number of correctly classified samples divided by total number of test
        samples.
    spiketrains : list of tuples
        Each entry in ``spiketrains`` contains a tuple ``(spiketimes, label)``
        for each layer of the network (for the last test sample only).
        ``spiketimes`` is a 2D array where the first index runs over the number
        of neurons in the layer, and the second index contains the spike times
        of the specific neuron.
        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'03Dense'``.
    vmem : list of tuples
        Each entry in ``vmem`` contains a tuple ``(vm, label)`` for each layer
        of the network (for the first test sample only). ``vm`` is a 2D array
        where the first index runs over the number of neurons in the layer, and
        the second index contains the membrane voltage of the specific neuron.
        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'03Dense'``.
    """
    from scipy.stats import pearsonr
    from snntoolbox.io.plotting import plot_layer_correlation, plot_potential
    from snntoolbox.io.plotting import plot_rates_minus_activations

    # Load neuron layers and connections if conversion was done during a
    # previous session.
    if 'snn_pyNN' in kwargs:
        layers = kwargs['snn_pyNN']
    elif 'snn_brian2' in kwargs:
        snn = kwargs['snn_brian2']
        for obj in snn.objects:
            if 'poissongroup' in obj.name and 'thresholder' not in obj.name:
                input_layer = obj
        labels = kwargs['labels']
        spikemonitors = kwargs['spikemonitors']
        statemonitors = kwargs['statemonitors']
        num_layers = len(labels)
    elif not snntoolbox._SIMULATOR == 'brian2':
        # Setup pyNN simulator if it was not passed on from a previous session.
        # From the pyNN documentation:
        # "Before using any other functions or classes from PyNN, the user
        # must call the setup() function. Calling setup() a second time
        # resets the simulator entirely, destroying any network that may
        # have been created in the meantime."
        sim.setup(timestep=simparams['dt'])
        layers = load_assembly()
    if snntoolbox._SIMULATOR == 'brian2':
        namespace = {'v_thresh': cellparams['v_thresh'] * sim.volt,
                     'v_reset': cellparams['v_reset'] * sim.volt,
                     'tau_m': cellparams['tau_m'] * sim.ms}
    else:
        num_layers = len(layers)
        if globalparams['verbose'] > 1:
            echo("Restoring layer connections...\n")
        for i in range(num_layers-1):
            filename = os.path.join(globalparams['path'],
                                    layers[i].label + '_' + layers[i+1].label)
            assert os.path.isfile(filename), \
                "Connections were not found at specified location.\n"
            # Turn off warning because we have no influence on it:
            # "UserWarning: ConvergentConnect is deprecated and will be
            # removed in a future version of NEST. Please use Connect
            # instead!"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                warnings.warn('deprecated', UserWarning)
                # Connect layers
                sim.Projection(layers[i], layers[i+1],
                               sim.FromFileConnector(filename))
        # Set cellparameters of neurons in each layer and initialize membrane
        # potential.
        for layer in layers[1:]:
            layer.set(**cellparams)
            layer.initialize(v=layers[1].get('v_rest'))
        # The spikes of the last layer are recorded by default because they
        # contain the networks output (classification guess).
        layers[-1].record(['spikes'])

    results = []

    # Iterate over the number of samples to test
    for test_num in range(simparams['num_to_test']):
        # Specify variables to record. For performance reasons, record spikes
        # and potential only for the last test sample. Have to reload network
        # in order to tell the layers to record new variables.
        if globalparams['verbose'] > 1 and \
                test_num == simparams['num_to_test'] - 1 and \
                not snntoolbox._SIMULATOR == 'brian2':
            if simparams['num_to_test'] > 1:
                echo("For last run, record spike rates and membrane " +
                     "potential of all layers.\n")
                layers = load_assembly()
                for i in range(num_layers-1):
                    filename = os.path.join(globalparams['path'],
                                            layers[i].label + '_' +
                                            layers[i+1].label)
                    # Turn off warning because we have no influence on it:
                    # "UserWarning: ConvergentConnect is deprecated and
                    # will be removed in a future version of NEST. Please
                    # use Connect instead!"
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        warnings.warn('deprecated', UserWarning)
                        # Connect layers
                        sim.Projection(layers[i], layers[i+1],
                                       sim.FromFileConnector(filename))
            layers[0].record(['spikes'])
            for layer in layers[1:]:
                layer.set(**cellparams)
                layer.initialize(v=layers[1].get('v_rest'))
                if globalparams['verbose'] == 3:
                    layer.record(['spikes', 'v'])
                else:
                    layer.record(['spikes'])

        # Pick a random test sample from among all possible input samples
        ind = randint(0, len(X_test) - 1)

        # Add Poisson input.
        if globalparams['verbose'] > 1:
            echo("Creating poisson input...\n")
        if snntoolbox._SIMULATOR == 'brian2':
            input_layer.rates = X_test[ind, :].flatten() * \
                simparams['max_f'] * sim.Hz
        else:
            rates = X_test[ind, :].flatten() * simparams['max_f']
            layers[0].set(duration=simparams['duration'])
            for (i, ss) in enumerate(layers[0]):
                ss.rate = rates[i]

        # Run simulation for 'duration'.
        if globalparams['verbose'] > 1:
            echo("Starting new simulation...\n")
        if snntoolbox._SIMULATOR == 'brian2':
            snn.store()
            snn.run(simparams['duration'] * sim.ms, namespace=namespace)
        else:
            sim.run(simparams['duration'])

        # Get result by comparing the guessed class (i.e. the index of the
        # neuron in the last layer which spiked most) to the ground truth.
        if snntoolbox._SIMULATOR == 'brian2':
            guesses = np.argmax(spikemonitors[-1].count)
        else:
            tmp = []
            for spiketrain in layers[-1].get_data().segments[-1].spiketrains:
                tmp.append(len(spiketrain))
            guesses = np.argmax(tmp)
        truth = np.argmax(Y_test[ind, :])
        results.append(guesses == truth)
        if globalparams['verbose'] > 0:
            echo("Sample {} of {} completed.\n".format(test_num + 1,
                 simparams['num_to_test']))
            echo("Moving average accuracy: {:.2%}.\n".format(np.mean(results)))

        # Reset simulation time and recorded network variables for next run.
        if globalparams['verbose'] > 1:
            echo("Resetting simulator...\n")
        if snntoolbox._SIMULATOR == 'brian2':
            # Skip during last run so the recorded variables are not discarded
            if test_num < simparams['num_to_test'] - 1:
                snn.restore()
        else:
            sim.reset()
        if globalparams['verbose'] > 1:
            echo("Done.\n")

    # Plot spikerates and spiketrains of layers. To visualize the spikerates,
    # neurons in hidden layers are spatially arranged on a 2d rectangular grod,
    # and the firing rate of each neuron on the grid is encoded by color.
    # Also plot the membrane potential vs time (except for the input layer).
    if globalparams['verbose'] > 1:
        echo("Simulation finished. Collecting results and saving plots...\n")
        # Turn off warning because we have no influence on it:
        # "FutureWarning: elementwise comparison failed; returning scalar
        #  instead, but in the future will perform elementwise comparison"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            warnings.warn('deprecated', FutureWarning)

            # Collect spiketrains of all layers, for the last test sample.
            spiketrains = []
            spikerates = []
            for i in range(num_layers):
                sp = []
                if snntoolbox._SIMULATOR == 'brian2':
                    for j in range(len(spikemonitors[i].spike_trains())):
                        sp.append(np.array(
                                spikemonitors[i].spike_trains()[j] / sim.ms))
                    label = labels[i]
                else:
                    # Iterate over neurons to convert Sequence to array
                    for spiketrain in layers[i].get_data().segments[-1].\
                                                                   spiketrains:
                        sp.append(np.array(spiketrain))
                    label = layers[i].label
                spiketrains.append((sp, label))
                # Count number of spikes fired in the layer and divide by the
                # simulation time in seconds to get the mean firing rate of
                # each neuron in Hertz.
                spikerates.append(([1000 * len(np.nonzero(neuron)[0]) /
                                    simparams['duration'] for neuron in sp],
                                   label))

            # Maybe repeat for membrane potential, skipping input layer
            vmem = []
            if globalparams['verbose'] == 3:
                for i in range(1, num_layers):
                    vm = []
                    if snntoolbox._SIMULATOR == 'brian2':
                        for v in statemonitors[i-1].v:
                            vm.append(np.array(v / 1e6 / sim.mV).transpose())
                        label = labels[i]
                    else:
                        for v in layers[i].get_data().segments[-1].\
                                                            analogsignalarrays:
                            vm.append(np.array(v))
                        label = layers[i].label
                    vmem.append((vm, label))
                if snntoolbox._SIMULATOR == 'brian2':
                    times = statemonitors[0].t / sim.ms
                else:
                    times = simparams['dt'] * np.arange(len(vmem[0][0][0]))

            # Load ANN and compute activations in each layer to compare with
            # SNN spikerates.
            echo("Calculating ANN layer activations " +
                 "to compare with SNN spike rates...\n")
            filename = 'ann_' + globalparams['filename']
            if os.path.isfile(os.path.join(globalparams['path'],
                                           filename+'_normWeights')):
                filename += '_normWeights'
            model = load_model(filename)['model']
            # Do this import here so the toolbox can be used without installing
            # Theano by setting globalparams['verbose'] < 2
            from snntoolbox.core.util import get_activations
            layer_activations = get_activations(model, X_test[ind])

            j = 0
            corr = []
            labels = []
            showLegend = False
            path = os.path.join(globalparams['path'], 'log', 'gui')
            for i in range(num_layers):
                sp = spikerates[i]
                newpath = os.path.join(path, sp[1])
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                plot_spiketrains(spiketrains[i], newpath)
                # Get layer shape from label and reshape the 1D SNN layer to
                # its original form.
                if i == 0 and globalparams['architecture'] == 'mlp':
                    if globalparams['dataset'] == 'mnist':
                        shape = (1, 28, 28)
                    elif globalparams['dataset'] == 'cifar10':
                        shape = (3, 32, 32)
                else:
                    shape = extract_label(sp[1])[2]
                rates_reshaped = np.reshape(sp[0], shape)
                plot_layer_activity((rates_reshaped, sp[1]),
                                    'Spikerates', newpath)
                if j < len(layer_activations):
                    plot_layer_activity(layer_activations[j], 'Activations',
                                        newpath)
                    j += 1
                plot_rates_minus_activations(rates_reshaped,
                                             layer_activations[j][0], sp[1],
                                             newpath)
                activations = layer_activations[j][0].flatten()
                (r, p) = pearsonr(rates, activations)
                corr.append(r)
                labels.append(sp[1])
                plot_layer_correlation(rates, activations, sp[1], newpath)
                if globalparams['verbose'] == 3:
                    if i == num_layers - 2:
                        showLegend = True
                    if i < num_layers - 1:
                        plot_potential(times, vmem[i], showLegend=showLegend)

    total_acc = np.mean(results)
    s = 's'
    if simparams['num_to_test'] == 1:
        s = ''
    echo("Total accuracy: {:.2%} on {} test sample{}.\n\n".format(
         total_acc, simparams['num_to_test'], s))

    if snntoolbox._SIMULATOR == 'brian2':
        snn.restore()
    else:
        sim.end()

    return total_acc


def run_SNN_keras(X_test, Y_test, **kwargs):
    """
    Simulate a spiking network with leaky integrate-and-fire units and
    Poisson input, using mean pooling and a timestepped approach.

    If ``globalparams['verbose'] > 1``, the simulator plots the spiketrains
    and spikerates of each neuron in each layer, for all
    ``globalparams['batch_size']`` samples of the first batch of ``X_test``.
    This is of course very costly in terms of memory and time, but can be
    useful for debugging the network's general functioning. To get detailed
    information about the network's behavior for a particular sample, replace
    the test set ``X_test, Y_test`` with this sample of interest.

    Parameters
    ----------

    X_test : float32 array
        The input samples to test.
        With data of the form (channels, num_rows, num_cols),
        X_test has dimension (num_samples, channels*num_rows*num_cols)
        for a multi-layer perceptron, and
        (num_samples, channels, num_rows, num_cols) for a convolutional net.
    Y_test : float32 array
        Ground truth of test data. Has dimension (num_samples, num_classes)
    snn : Keras model, possible kwarg
        The network architecture and weights in json format.
    get_output : Theano function, possible kwarg
        Gets the output of the network.

    Returns
    -------

    total_acc : float
        Number of correctly classified samples divided by total number of test
        samples.
    spiketrains : list of tuples
        Each entry in ``spiketrains`` contains a tuple ``(spiketimes, label)``
        for each layer of the network (for the first batch only).
        ``spiketimes`` is a 2D array where the first index runs over the number
        of neurons in the layer, and the second index contains the spike times
        of the specific neuron.
        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'03Dense'``.

    """

    # Load neuron layers and connections if conversion was done during a
    # previous session.
    if 'snn_keras' in kwargs:
        snn = kwargs['snn_keras']
        get_output = kwargs['get_output']
    else:
        if globalparams['verbose'] > 1:
            echo("Restoring layer connections...\n")
        model_dict = load_model('snn_keras_' + globalparams['filename'],
                                spiking=True)
        snn = model_dict['model']
        get_output = model_dict['get_output']

    # Ground truth
    truth = np.argmax(Y_test, axis=1)

    # This factor determines the probability threshold for cells in the input
    # layer to fire a spike. Increasing ``max_f`` increases the firing rate.
    rescale_fac = 1000 / (simparams['max_f'] * simparams['dt'])

    # Divide the test set into batches and run all samples in a batch in
    # parallel.
    output = np.zeros(Y_test.shape).astype('int32')
    num_batches = int(np.ceil(X_test.shape[0] / globalparams['batch_size']))
    if globalparams['verbose'] > 1:
        echo("Starting new simulation...\n")
    # Allocate a list 'spiketrains' with the following specification:
    # Each entry in ``spiketrains`` contains a tuple ``(spiketimes, label)``
    # for each layer of the network (for the first batch only).
    # ``spiketimes`` is a 2D array where the first index runs over the number
    # of neurons in the layer, and the second index contains the spike times
    # of the specific neuron.
    # ``label`` is a string specifying both the layer type and the index,
    # e.g. ``'03Dense'``.
    spiketrains = []
    for layer in snn.layers:
        shape = (int(np.prod(layer.output_shape[1:])),
                 int(simparams['duration'] / simparams['dt']))
        spiketrains.append((np.zeros(shape), layer.name))

    from copy import deepcopy
    # Allocate a list to store the spiketrains lists for each sample in the
    # batch, as specified above.
    spiketrains_batch = [deepcopy(spiketrains) for sample in
                         range(globalparams['batch_size'])]
    for batch_idx in range(num_batches):
        # Determine batch indices.
        max_idx = min((batch_idx + 1) * globalparams['batch_size'],
                      Y_test.shape[0])
        batch_idxs = range(batch_idx * globalparams['batch_size'], max_idx)
        batch = X_test[batch_idxs, :]

        # Reset network variables.
        sim.reset(snn.layers[-1])

        # Loop through simulation time.
        t_idx = 0
        for t in np.arange(0, simparams['duration'], simparams['dt']):
            # Create poisson input.
            spike_snapshot = np.random.random_sample(batch.shape) * rescale_fac
            inp_images = (spike_snapshot <= batch).astype('float32')
            # Main step: Propagate poisson input through network and record
            # output spikes.
            out_spikes, ts = get_output(inp_images, float(t))
            # For the first batch only, record the spiketrains of each neuron
            # in each layer.
            if batch_idx == 0 and globalparams['verbose'] > 1:
                for layer in range(len(snn.layers)):
                    # s is an array of dimension
                    # (batch_size, num_neurons_in_layer)
                    s = snn.layers[layer].spiketrain.get_value()
                    # Split s to separate the spike times of each sample in the
                    # batch.
                    for sample in range(globalparams['batch_size']):
                        spiketrains_batch[sample][layer][0][:, t_idx] = \
                            s[sample, :].flatten()
            t_idx += 1
            # Count number of spikes in output layer during whole simulation.
            output[batch_idxs, :] += out_spikes.astype('int32')
            if globalparams['verbose'] > 1:
                echo('.')
        if globalparams['verbose'] > 1:
            # Get result by comparing the guessed class (i.e. the index of the
            # neuron in the last layer which spiked most) to the ground truth.
            guesses = np.argmax(output, axis=1)
            echo('\n')
            echo("Batch {} of {} completed ({:.1%})\n".format(batch_idx + 1,
                 num_batches, (batch_idx + 1) / num_batches))
            echo("Moving average accuracy: {:.2%}.\n".format(
                  np.sum(guesses[:max_idx] == truth[:max_idx]) / max_idx))
            if batch_idx == 0 and globalparams['verbose'] > 2:
                echo('\n')
                echo("Saving plots for last image in first batch...\n")
                path = os.path.join(globalparams['path'], 'log', 'gui')
                plot_layer_summaries(snn, spiketrains_batch[-1],
                                     X_test[globalparams['batch_size'] - 1],
                                     path)
                plot_pearson_coefficients(spiketrains_batch, batch, path)

    guesses = np.argmax(output, axis=1)
    total_acc = np.mean(guesses == truth)
    echo("Simulation finished.\n\n")
    echo("Total output spikes: {}\n".format(np.sum(output)))
    echo("Average output spikes per sample: {:.2f}\n".format(
         np.mean(np.sum(output, axis=1))))
    echo("Final time: {:.3f}\n".format(ts.max()))
    echo("Total accuracy: {:.2%} on {} test samples.\n\n".format(total_acc,
         output.shape[0]))

    return total_acc
