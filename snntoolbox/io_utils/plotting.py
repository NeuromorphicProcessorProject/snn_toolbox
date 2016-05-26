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

from snntoolbox.config import settings
from snntoolbox.core.util import wilson_score

standard_library.install_aliases()


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

    shape = [np.prod(layer[0].shape[:-1]), layer[0].shape[-1]]
    spiketrains = np.reshape(layer[0], shape)
    plt.figure()
    # Iterate over neurons in layer
    for (neuron, spiketrain) in enumerate(spiketrains):
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
    plt.gca().set_xlim(0, settings['duration'])
    plt.gca().set_ylim([-0.1, neuron + 1])
    # Replace 'name' by 'layer[1]' to get more info into the title
    # name = extract_label(layer[1])[1]
    name = layer[1]
    plt.title('Spiketrains \n of layer {}'.format(name))
    plt.xlabel('time [ms]')
    plt.ylabel('neuron index')
    if path is not None:
        filename = '7Spiketrains'
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
    # Plot feature maps as 2d-images next to each other, but start a new column
    # when four rows are filled, since the number of features in a
    # convolution layer is often a multiple of 4.
    else:
        if num < 4:  # Arrange less than 4 feature maps in a single row.
            num_rows = 1
            num_cols = num
        else:  # Arrange more than 4 feature maps in a rectangle.
            num_rows = 4
            num_cols = int(np.ceil(num / num_rows))
        if num_cols > 4:
            fac = num_cols / 4
        f, ax = plt.subplots(num_rows, num_cols, squeeze=False,
                             figsize=(3 + num_cols * 2, 11))
        for i in range(num_rows):
            for j in range(num_cols):
                idx = j + num_cols * i
                if idx >= num:
                    break
                im = ax[i, j].imshow(layer[0][idx], interpolation='nearest',
                                     clim=limits)
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)
    # Replace 'name' by 'layer[1]' to get more info into the title
    # name = extract_label(layer[1])[1]
    name = layer[1]
    unit = ' [Hz]' if title == 'Spikerates' else ''
    f.suptitle('{} \n of layer {}'.format(title + unit, name), fontsize=20)
    f.subplots_adjust(left=0, bottom=0, right=0.99, top=0.9,
                      wspace=0.05, hspace=0.05)
    cax = f.add_axes([0.99, 0, 0.05 / fac, 0.99])
    cax.locator_params(nbins=8)
    f.colorbar(im, cax=cax, orientation='vertical')
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
    plt.close()


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
    plt.plot(times, np.ones_like(times) * settings['v_thresh'], 'r--',
             label='V_thresh')
    plt.plot(times, np.ones_like(times) * settings['v_reset'], 'b-.',
             label='V_reset')
    plt.ylim([settings['v_reset'] - 0.1, settings['v_thresh'] + 0.1])
    if showLegend:
        plt.legend(loc='upper left', prop={'size': 15})
    plt.xlabel('Time [ms]')
    plt.ylabel('Membrane potential')
    # Replace 'name' by 'layer[1]' to get more info into the title
    # name = extract_label(layer[1])[1]
    name = layer[1]
    plt.title('Membrane potential for neurons \n in layer {}\n'.format(name))
    if path is not None:
        filename = 'Potential'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_layer_summaries(spiketrains, spikerates, activations, path=None):
    # Loop over layers
    for i in range(len(spiketrains)):
        label = spiketrains[i][1]
        newpath = os.path.join(path, label)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        plot_spiketrains(spiketrains[i], newpath)
        plot_layer_activity(spikerates[i], 'Spikerates', newpath)
        plot_layer_activity(activations[i], 'Activations', newpath)
        plot_rates_minus_activations(spikerates[i][0], activations[i][0],
                                     label, newpath)
        plot_layer_correlation(spikerates[i][0].flatten(),
                               activations[i][0].flatten(),
                               'ANN-SNN correlations\n of layer ' + label,
                               newpath)
        plot_hist({'Spikerates': spikerates[i][0].flatten()}, 'Spikerates',
                  label, newpath)


def plot_activity_distribution(activity_dict, path=None):
    h = {}
    for (key, val) in activity_dict.items():
        l = []
        for a in val:
            l += list(a[0].flatten())
        h.update({key: l})
    plot_hist_combined(h, path=path)


def plot_hist_combined(h, path=None):
    keys = list(h.keys())
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', which='both')
    ax.get_xaxis().set_visible(False)
    axes = [ax]
    fig.subplots_adjust(top=0.8)
    colors = plt.cm.spectral(np.linspace(0, 0.9, len(keys)))
    for i in range(len(keys)):
        axes.append(ax.twiny())
        axes[-1].hist(h[keys[i]], label=keys[i], color=colors[i], log=True,
                      histtype='step', bottom=1)
        unit = ' [Hz]' if keys[i] == 'Spikerates' else ''
        axes[-1].set_xlabel(keys[i] + unit, color=colors[i])
        axes[-1].tick_params(axis='x', colors=colors[i])
        if i > 0:
            axes[-1].set_frame_on(True)
            axes[-1].patch.set_visible(False)
            axes[-1].xaxis.set_ticks_position('bottom')  # 1-i/10
            axes[-1].xaxis.set_label_position('bottom')
    plt.title('Distribution', y=1.15)
    filename = 'Activity_distribution'
    if path:
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_hist(h, title=None, layer_label=None, path=None, scale_fac=None,
              applied_fac=None):
    keys = sorted(h.keys())
    plt.hist([h[key] for key in keys], label=keys, log=True, bottom=1)
    plt.legend()
    if title and layer_label:
        if 'Spikerates' in title:
            filename = '4' + title + '_distribution'
            unit = '[Hz]'
        else:
            filename = layer_label + '_' + title + '_distribution'
            unit = ''
        if scale_fac and applied_fac:
            facs = "Max (act, W): {:.2f}; applied divisor: {:.2f}".format(
                scale_fac, applied_fac)
        else:
            facs = ''
        plt.title('{} distribution {} \n of layer {} \n {}'.format(
            title, unit, layer_label, facs))
    else:
        plt.title('Distribution')
        filename = 'Activity_distribution'
    if path:
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_pearson_coefficients(spikerates_batch, activations_batch, path=None):
    """
    Plot the Pearson correlation coefficients for each layer, averaged over one
    mini batch.
    """
    from scipy.stats import pearsonr

    batch_size = len(spikerates_batch[0][0])

    co = []
    j = 0
    for layer_num in range(len(spikerates_batch)):
        if 'Flatten' not in spikerates_batch[layer_num][1]:
            c = []
            for sample in range(batch_size):
                (r, p) = pearsonr(
                    spikerates_batch[layer_num][0][sample].flatten(),
                    activations_batch[layer_num][0][sample].flatten())
                c.append(r)
            j += 1
            co.append(c)

    # Average over batch
    corr = np.mean(co, axis=1)
    std = np.std(co, axis=1)

    labels = [sp[1] for sp in spikerates_batch if 'Flatten' not in sp[1]]

    plt.figure()
    plt.bar([i + 0.1 for i in range(len(corr))], corr, width=0.8, yerr=std,
            color='#f5f5f5')
    plt.ylim([0, 1])
    plt.xlim([0, len(corr)])
    plt.xticks([i + 0.5 for i in range(len(labels))], labels, rotation=90)
    plt.tick_params(bottom='off')
    plt.title('Correlation between ANN activations \n and SNN spikerates,\n ' +
              'averaged over {} samples'.format(batch_size))
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
    plt.xlabel('SNN spikerates [Hz]', fontsize=16)
    plt.ylabel('ANN activations', fontsize=16)
    if path is not None:
        filename = '5Correlation'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_rates_minus_activations(rates, activations, label, path=None):
    rates_norm = rates / np.max(rates)
    activations_norm = activations / np.max(activations)
    plot_layer_activity((rates_norm - activations_norm, label),
                        'Spikerates_minus_Activations', path, limits=(-1, 1))


def output_graphs(spiketrains_batch, ann, batch, path=None):
    from snntoolbox.core.util import get_activations_batch
    from snntoolbox.core.util import spiketrains_to_rates
    from snntoolbox.core.util import get_sample_activity_from_batch

    print('\n')
    print("Saving plots of one sample to {}...\n".format(path))

    spiketrains = get_sample_activity_from_batch(spiketrains_batch)
    spikerates_batch = spiketrains_to_rates(spiketrains_batch)
    spikerates = get_sample_activity_from_batch(spikerates_batch)
    activations_batch = get_activations_batch(ann, batch)
    activations = get_sample_activity_from_batch(activations_batch)

    plot_layer_summaries(spiketrains, spikerates, activations, path)
    plot_pearson_coefficients(spikerates_batch, activations_batch, path)
    plot_activity_distribution({'Spikerates': spikerates_batch,
                                'Activations': activations_batch}, path)
    print("Done.\n")


def plot_param_sweep(results, n, params, param_name, param_logscale):
    # Compute confidence intervals of the experiments
    ci = [wilson_score(q, n) for q in results]
    ax = plt.subplot()
    if param_logscale:
        ax.set_xscale('log', nonposx='clip')
    ax.errorbar(params, results, yerr=ci, fmt='x-')
    ax.set_title('Accuracy vs Hyperparameter')
    ax.set_xlabel(param_name)
    ax.set_ylabel('accuracy')
    fac = 0.9
    if params[0] < 0:
        fac += 0.2
    ax.set_xlim(fac * params[0], 1.1 * params[-1])
    ax.set_ylim(0, 1)


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
