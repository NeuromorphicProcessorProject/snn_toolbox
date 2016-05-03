# -*- coding: utf-8 -*-
"""
Various functions to visualize connectivity, activity and accuracy of the
network.

Created on Wed Nov 18 13:57:37 2015

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import os
import numpy as np
import matplotlib.pyplot as plt

from snntoolbox.config import cellparams, simparams
# from snntoolbox.core.util import extract_label

standard_library.install_aliases()

plt.ion()


def plot_spiketrains(layer, path=None):
    """
    Plot which neuron fired at what time during the simulation.

    Parameters
    ----------

    layer : tuple
        ``(spiketimes, label)``.

        ``spiketimes`` is a 2D array where the first index runs over the number
        of neurons in the layer, and the second index contains the spike times
        of the specific neuron.

        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'3Dense'``.

    path : string, optional
        If not none, specifies where to save the resulting image.
    """

    plt.figure()
    # Iterate over neurons in layer
    for (neuron, spiketrain) in enumerate(layer[0]):
        # Remove zeros from spiketrain which falsely indicate spikes at time 0.
        # Spikes at time 0 are forbidden (and indeed prevented in the
        # simulation), because of this difficulty to distinguish them from a 0
        # entry indicating no spike.
        if 0 in spiketrain:
            spikelist = [j for j in spiketrain if j != 0]
            # Create an array of the same size as the spikelist containing just
            # the neuron index.
            y = np.ones_like(spikelist) * neuron
            plt.plot(spikelist, y, '.')
        else:
            y = np.ones_like(spiketrain) * neuron
            plt.plot(spiketrain, y, '.')
    plt.gca().set_xlim(0, simparams['duration'])
    plt.gca().set_ylim([-0.1, neuron + 1])
    # Replace 'name' by 'layer[1]' to get more info into the title
    # name = extract_label(layer[1])[1]
    name = layer[1]
    plt.title('Spiketrains \n of layer {}'.format(name))
    plt.xlabel('time [ms]')
    plt.ylabel('neuron index')
    if path is not None:
        filename = '4Spiketrains'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_layer_activity(layer, title, path=None, limits=None):
    """
    Visualize a layer by arranging the neurons in a line or on a 2D grid.

    Can be used to show average firing rates of individual neurons in an SNN,
    or the activation function per layer in an ANN.
    The activity is encoded by color.

    Parameters
    ----------

    layer : tuple
        ``(activity, label)``.

        ``activity`` is an array of the same shape as the original layer,
        containing e.g. the spikerates or activations of neurons in a layer.

        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'3Dense'``.

    title : string
        Figure title.

    path : string, optional
        If not ``None``, specifies where to save the resulting image.

    limits : tuple, optional
        If not ``None``, the colormap of the resulting image is limited by this
        tuple.
    """

    # Highest possible spike rate, used to normalize image plots. Need to
    # define these here because we plot only one colorbar for several
    # subplots, each having a possibly different range.
    vmax = np.max(layer[0])
    vmin = np.min(layer[0])

    if limits is None:
        limits = (vmin, vmax)

    shape = layer[0].shape
    num = shape[0]
    fac = 1  # Scales height of colorbar
    # Case: One-dimensional layer (e.g. Dense). If larger than 100 neurons,
    # form a rectangle. Otherwise plot a 1d-image.
    if len(shape) == 1:
        if num >= 100:
            n = int(np.sqrt(num))
            while num / n % 1 != 0:
                n -= 1
            f, ax = plt.subplots(figsize=(7, min(3 + n*n*n / num / 2, 9)))
            im = ax.imshow(np.reshape(layer[0], (n, -1)),
                           interpolation='nearest', clim=limits)
        else:
            f, ax = plt.subplots(figsize=(7, 2))
            im = ax.imshow(np.array(layer[0], ndmin=2),
                           interpolation='nearest', clim=limits)
        ax.get_yaxis().set_visible(False)
    # Case: Multi-dimensional layer, where first dimension gives the number of
    # channels (input layer) or feature maps (convolution layer).
    # Plot feature maps as 2d-images next to each other, but start a new row
    # when four columns are filled, since the number of features in a
    # convolution layer is usually a multiple of 4.
    else:
        if num < 4:
            num_cols = num
        else:
            num_cols = 4
        num_rows = int(np.ceil(num / num_cols))
        if num_rows > 4:
            fac = num_rows / 4
        f, ax = plt.subplots(num_rows, num_cols, figsize=(7, 2 + num_rows * 2),
                             squeeze=False)
        for i in range(num_rows):
            for j in range(num_cols):
                idx = j + num_cols * i
                if idx >= num:
                    break
                im = ax[i, j].imshow(layer[0][idx],
                                     interpolation='nearest', clim=limits)
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)
    # Replace 'name' by 'layer[1]' to get more info into the title
    # name = extract_label(layer[1])[1]
    name = layer[1]
    f.suptitle('{} \n of layer {}'.format(title, name), fontsize=20)
    f.subplots_adjust(left=0, bottom=0.072, right=1, top=0.9,
                      wspace=0.05, hspace=0.05)
    cax = f.add_axes([0.05, 0, 0.9, 0.05 / fac])
    cax.locator_params(nbins=8)
    f.colorbar(im, cax=cax, orientation='horizontal')
    if path is not None:
        if title == 'Activations':
            filename = '0' + title
        elif title == 'Spikerates':
            filename = '1' + title
        elif title == 'Spikerates_minus_Activations':
            filename = '2' + title
        else:
            filename = title
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        f.show()


def plot_correlations(spikerates, layer_activations):
    """
    Plot the correlation between SNN spiketrains and ANN activations.

    For each layer, the method draws a scatter plot, showing the correlation
    between the average firing rate of neurons in the SNN layer and the
    activation of the corresponding neurons in the ANN layer.

    Parameters
    ----------

    spikerates : list of tuples ``(spikerate, label)``.

        ``spikerate`` is a 1D array containing the mean firing rates of the
        neurons in a specific layer.

        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'3Dense'``.

    layer_activations : list of tuples ``(activations, label)``
        Each entry represents a layer in the ANN for which an activation can be
        calculated (e.g. ``Dense``, ``Convolution2D``).

        ``activations`` is an array of the same dimension as the corresponding
        layer, containing the activations of Dense or Convolution layers.

        ``label`` is a string specifying the layer type, e.g. ``'Dense'``.
    """

    rates = []
    labels = []
    num_layers = len(layer_activations)
    for layer in spikerates:
        # Replace 'name' by 'layer[1]' to get more info into the title
        # name = extract_label(layer[1])[1]
        name = layer[1]
        label = name
        rates.append(layer[0])
        labels.append(label)
    # Determine optimal shape for rectangular arrangement of plots
    num_rows = int(np.ceil(np.sqrt(num_layers)))
    num_cols = int(np.ceil(num_layers / num_rows))
    f, ax = plt.subplots(num_rows, num_cols, figsize=(8, 1 + num_rows * 4),
                         squeeze=False)
    for i in range(num_rows):
        for j in range(num_cols):
            layer_num = j + i * num_cols
            if layer_num >= num_layers:
                break
            ax[i, j].plot(rates[layer_num],
                          layer_activations[layer_num][0].flatten(), '.')
            ax[i, j].set_title(labels[layer_num], fontsize='medium')
            ax[i, j].locator_params(nbins=4)
            ax[i, j].set_xlim([-max(rates[layer_num]) / 1000,
                               max(rates[layer_num]) * 1.1])
            ax[i, j].set_ylim([None,
                               np.max(layer_activations[layer_num][0]) * 1.1])
    f.suptitle('ANN-SNN correlations', fontsize=20)
    f.subplots_adjust(wspace=0.3, hspace=0.3)
    f.text(0.5, 0.04, 'SNN spikerates (Hz)', ha='center', fontsize=16)
    f.text(0.04, 0.5, 'ANN activations', va='center', rotation='vertical',
           fontsize=16)


def plot_potential(times, layer, showLegend=False, path=None):
    """
    Plot the membrane potential of a layer.

    Parameters
    ----------

    times : 1D array
        The time values where the potential was sampled.

    layer : tuple
        ``(vmem, label)``.

        ``vmem`` is a 2D array where the first index runs over the number of
        neurons in the layer, and the second index contains the membrane
        potential of the specific neuron.

        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'3Dense'``.

    showLegend : boolean, optional
        If ``True``, shows the legend indicating the neuron indices and lines
        like ``v_thresh``, ``v_rest``, ``v_reset``. Recommended only for layers
        with few neurons.

    path : string, optional
        If not none, specifies where to save the resulting image.
    """

    plt.figure()
    # Transpose layer array to get slices of vmem values for each neuron.
    for (neuron, vmem) in enumerate(layer[0]):
        plt.plot(times, vmem)
    plt.plot(times, np.ones_like(times) * cellparams['v_thresh'], 'r--',
             label='V_thresh')
    plt.plot(times, np.ones_like(times) * cellparams['v_reset'], 'b-.',
             label='V_reset')
    plt.ylim([cellparams['v_reset'] - 0.1, cellparams['v_thresh'] + 0.1])
    if showLegend:
        plt.legend(loc='upper left', prop={'size': 15})
    plt.xlabel('Time [ms]')
    plt.ylabel('Membrane potential')
    # Replace 'name' by 'layer[1]' to get more info into the title
    # name = extract_label(layer[1])[1]
    name = layer[1]
    plt.title('Membrane potential for neurons \n ' +
              'in layer {}\n'.format(name))
    if path is not None:
        filename = 'Potential'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_layer_summaries(snn, spiketrains, X, path=None):
    import os
    from snntoolbox.config import globalparams
    from snntoolbox.io.load import load_model
    from snntoolbox.core.util import get_activations

    # Load ANN and compute activations in each layer to compare
    # with SNN spikerates.
    filename = 'ann_' + globalparams['filename']
    if os.path.isfile(os.path.join(globalparams['path'],
                                   filename+'_normWeights.h5')):
        filename += '_normWeights'
    model = load_model(filename)['model']

    # Get the last sample of batch to plot results:
    layer_activations = get_activations(model, X)

    # Loop over layers
    j = 0
    spikerates = []
    labels = []
    for (i, sp) in enumerate(spiketrains):
        if 'Flatten' not in sp[1]:
            newpath = os.path.join(path, sp[1])
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            plot_spiketrains(sp, newpath)
            # Count number of spikes fired in the layer and divide
            # by the simulation time in seconds to get the mean
            # firing rate of each neuron in Hertz.
            spikerates.append(([1000 * len(np.nonzero(neuron)[0]) /
                                simparams['duration'] for neuron
                                in sp[0]], sp[1]))
            # Get layer shape from label and reshape the 1D SNN
            # layer to its original form.
            shape = snn.layers[i].output_shape[1:]
            rates_reshaped = np.reshape(np.array(spikerates[-1][0]), shape)
            plot_layer_activity((rates_reshaped, sp[1]), 'Spikerates',
                                newpath)
            plot_layer_activity(layer_activations[j], 'Activations', newpath)
            plot_rates_minus_activations(rates_reshaped,
                                         layer_activations[j][0], sp[1],
                                         newpath)
            labels.append(sp[1])
            title = 'ANN-SNN correlations\n of layer ' + sp[1]
            plot_layer_correlation(spikerates[-1][0],
                                   layer_activations[j][0].flatten(),
                                   title, newpath)
            j += 1

    print("Saved plots to {}.\n".format(path))


def plot_pearson_coefficients(spiketrains_batch, X_batch, path=None):
    """
    Plot the Pearson correlation coefficients for each layer, averaged over one
    mini batch.
    """
    from snntoolbox.config import globalparams
    from snntoolbox.io.load import load_model
    from snntoolbox.core.util import get_activations_batch
    from scipy.stats import pearsonr

    # Load ANN and compute activations in each layer to compare
    # with SNN spikerates.
    filename = 'ann_' + globalparams['filename']
    if os.path.isfile(os.path.join(globalparams['path'],
                                   filename+'_normWeights.h5')):
        filename += '_normWeights'
    model = load_model(filename)['model']

    spikerates = []
    co = []
    layer_activations = get_activations_batch(model, X_batch)
    for l in range(len(X_batch)):
        j = 0
        c = []
        for (i, sp) in enumerate(spiketrains_batch[l]):
            if 'Flatten' not in sp[1]:
                # Count number of spikes fired in the layer and divide
                # by the simulation time in seconds to get the mean
                # firing rate of each neuron in Hertz.
                spikerates.append(([1000 * len(np.nonzero(neuron)[0]) /
                                    simparams['duration'] for neuron
                                    in sp[0]], sp[1]))
                # Get layer shape from label and reshape the 1D SNN
                # layer to its original form.
                (r, p) = pearsonr(spikerates[-1][0],
                                  layer_activations[j][0][l].flatten())
                c.append(r)
                j += 1
        co.append(c)
    # Average over batch
    corr = np.mean(co, axis=0)
    std = np.std(co, axis=0)

    labels = [sp[1] for sp in spiketrains_batch[0] if 'Flatten' not in sp[1]]

    plt.figure()
    plt.bar([i + 0.1 for i in range(len(corr))], corr, width=0.8, yerr=std,
            color='#f5f5f5')
    plt.ylim([0, 1])
    plt.xlim([0, len(corr)])
    plt.xticks([i + 0.5 for i in range(len(labels))], labels, rotation=90)
    plt.tick_params(bottom='off')
    plt.title('Correlation between ANN activations \n and SNN spikerates,\n ' +
              'averaged over {} samples'.format(len(X_batch)))
    plt.ylabel('Pearson Correlation Coefficient')
    if path is not None:
        filename = 'Pearson'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_layer_correlation(rates, activations, title, path=None):
    plt.figure()
    plt.plot(rates, activations, '.')
    plt.title(title, fontsize=20)
    plt.locator_params(nbins=4)
    plt.xlim([-max(rates) / 1000, max(rates) * 1.1])
    plt.ylim([None, max(activations) * 1.1])
    plt.xlabel('SNN spikerates (Hz)', fontsize=16)
    plt.ylabel('ANN activations', fontsize=16)
    if path is not None:
        filename = '3Correlation'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_rates_minus_activations(rates, activations, label, path=None):
    rates_norm = rates / np.max(rates)
    activations_norm = activations / np.max(activations)
    plot_layer_activity((rates_norm - activations_norm, label),
                        'Spikerates_minus_Activations', path, limits=(-1, 1))


def plot_history(h):
    """
    Plot the training and validation loss and accuracy at each epoch.

    Parameters
    ----------

    h : Keras history object
        Contains the training and validation loss and accuracy at each epoch
        during training.
    """

    plt.figure

    plt.title('Accuracy and loss during training and validation')

    plt.subplot(211)
    plt.plot(h.history['acc'], label='acc')
    plt.plot(h.history['val_acc'], label='val_acc')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.grid(which='both')

    plt.subplot(212)
    plt.plot(h.history['loss'], label='loss')
    plt.plot(h.history['val_loss'], label='val_loss')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(which='both')

    plt.xlabel('epoch')
    plt.show()
