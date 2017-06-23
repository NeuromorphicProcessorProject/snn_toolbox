# -*- coding: utf-8 -*-
"""
Various functions to visualize connectivity, activity and accuracy of the
network.

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
from future import standard_library

standard_library.install_aliases()


def output_graphs(plot_vars, config, path=None, idx=0):
    """Wrapper function to display / save a number of plots.

    Parameters
    ----------

    plot_vars: dict
        Example items:

        - spiketrains_n_b_l_t: list[tuple[np.array, str]]
            Each entry in ``spiketrains_batch`` contains a tuple
            ``(spiketimes, label)`` for each layer of the network (for the first
            batch only, and excluding ``Flatten`` layers).
            ``spiketimes`` is an array where the last index contains the spike
            times of the specific neuron, and the first indices run over the
            number of neurons in the layer:
            (batch_size, n_chnls, n_rows, n_cols, duration)
            ``label`` is a string specifying both the layer type and the index,
            e.g. ``'03Dense'``.

        - activations_n_b_l: list[tuple[np.array, str]]
            Activations of the ANN.

        - spikecounts_n_b_l: list[tuple[np.array, str]]
            Spikecounts of the SNN. Used to compute spikerates.

    config: configparser.ConfigParser
        Settings.

    path: Optional[str]
        If not ``None``, specifies where to save the resulting image. Else,
        display plots without saving.

    idx: int
        The index of the sample to display. Defaults to 0.
    """

    from snntoolbox.simulation.utils import spiketrains_to_rates
    from snntoolbox.simulation.utils import get_sample_activity_from_batch

    if plot_vars == {}:
        return

    if path is not None:
        print("Saving plots of one sample to {}...\n".format(path))

    plot_keys = eval(config['output']['plot_vars'])
    duration = config.getint('simulation', 'duration')

    if 'activations_n_b_l' in plot_vars:
        plot_vars['activations_n_l'] = get_sample_activity_from_batch(
            plot_vars['activations_n_b_l'], idx)
    if 'spiketrains_n_b_l_t' in plot_vars:
        plot_vars['spiketrains_n_l_t'] = get_sample_activity_from_batch(
            plot_vars['spiketrains_n_b_l_t'], idx)
        if any({'spikerates', 'correlation', 'hist_spikerates_activations'}
               & plot_keys):
            plot_vars['spikerates_n_b_l'] = spiketrains_to_rates(
                plot_vars['spiketrains_n_b_l_t'], duration)
            plot_vars['spikerates_n_l'] = get_sample_activity_from_batch(
                plot_vars['spikerates_n_b_l'], idx)

    plot_layer_summaries(plot_vars, config, path)

    print("Plotting batch run statistics...")
    if 'spikecounts' in plot_keys:
        plot_spikecount_vs_time(plot_vars['spiketrains_n_b_l_t'], duration,
                                config.getfloat('simulation', 'dt'), path)
    if 'correlation' in plot_keys:
        plot_pearson_coefficients(plot_vars['spikerates_n_b_l'],
                                  plot_vars['activations_n_b_l'], config, path)
    if 'hist_spikerates_activations' in plot_keys:
        s = a = []
        for ss, aa in zip(plot_vars['spikerates_n_b_l'],
                          plot_vars['activations_n_b_l']):
            s += list(np.divide(np.ndarray.flatten(ss[0]), 1000.))
            a += list(np.ndarray.flatten(aa[0]))
        plot_hist({'Spikerates': s, 'Activations': a}, path=path)
    print("Done.\n")


def plot_layer_summaries(plot_vars, config, path=None):
    """Display or save a number of plots for a specific layer.

    Parameters
    ----------

    plot_vars: dict

        Example items:

        - spikerates: list[tuple[np.array, str]]
            Each entry in ``spikerates`` contains a tuple ``(rates, label)`` for
            each layer of the network (for the first batch only, and excluding
            ``Flatten`` layers).

            ``rates`` contains the average firing rates of all neurons in a
            layer. It has the same shape as the original layer, e.g.
            (n_features, n_rows, n_cols) for a convolution layer.

            ``label`` is a string specifying both the layer type and the index,
            e.g. ``'03Dense'``.

        - activations: list[tuple[np.array, str]]
            Contains the activations of a net. Same structure as ``spikerates``.

        - spiketrains: list[tuple[np.array, str]]
            Each entry in ``spiketrains`` contains a tuple
            ``(spiketimes, label)`` for each layer of the network (for the first
            batch only, and excluding ``Flatten`` layers).

            ``spiketimes`` is an array where the last index contains the spike
            times of the specific neuron, and the first indices run over the
            number of neurons in the layer: (n_chnls, n_rows, n_cols, duration)

            ``label`` is a string specifying both the layer type and the index,
            e.g. ``'03Dense'``.

    config: configparser.ConfigParser
        Settings.

    path: Optional[str]
        If not ``None``, specifies where to save the resulting image. Else,
        display plots without saving.
    """

    from snntoolbox.utils.utils import extract_label

    plot_keys = eval(config['output']['plot_vars'])

    if len(plot_vars.keys()) == 0:
        return

    num_layers = len(list(plot_vars.values())[0])

    for i in range(num_layers):
        label = list(plot_vars.values())[0][i][1]
        name = extract_label(label)[1] \
            if config.getboolean('output', 'use_simple_labels') else label
        print("Plotting layer {}".format(label))
        newpath = os.path.join(path, label)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if 'spiketrains' in plot_keys:
            plot_spiketrains(plot_vars['spiketrains_n_l_t'][i], newpath)
        if 'spikerates' in plot_keys:
            plot_layer_activity(plot_vars['spikerates_n_l'][i],
                                str('Spikerates'), newpath)
            plot_hist(
                {'Spikerates': plot_vars['spikerates_n_l'][i][0].flatten()},
                'Spikerates', name, newpath)
        if 'activations' in plot_keys:
            plot_layer_activity(plot_vars['activations_n_l'][i],
                                str('Activations'), newpath)
        if 'spikerates_n_l' in plot_vars and 'activations_n_l' in plot_vars:
            plot_activations_minus_rates(plot_vars['spikerates_n_l'][i][0],
                                         plot_vars['activations_n_l'][i][0],
                                         name, newpath)
        if 'correlation' in plot_keys:
            plot_layer_correlation(plot_vars['spikerates_n_l'][i][0].flatten(),
                                   plot_vars['activations_n_l'][i][0].flatten(),
                                   str('ANN-SNN correlations\n of layer '+name),
                                   config, newpath)


def plot_layer_activity(layer, title, path=None, limits=None):
    """Visualize a layer by arranging the neurons in a line or on a 2D grid.

    Can be used to show average firing rates of individual neurons in an SNN,
    or the activation function per layer in an ANN.
    The activity is encoded by color.

    Parameters
    ----------

    layer: tuple[np.array, str]
        ``(activity, label)``.

        ``activity`` is an array of the same shape as the original layer,
        containing e.g. the spikerates or activations of neurons in a layer.

        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'3Dense'``.

    title: str
        Figure title.

    path: Optional[str]
        If not ``None``, specifies where to save the resulting image. Else,
        display plots without saving.

    limits: Optional[tuple]
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

    im = None
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
            f, ax = plt.subplots(figsize=(7, min(3+n*n*n/num/2, 9)))
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
                             figsize=(3+num_cols*2, 11))
        for i in range(num_rows):
            for j in range(num_cols):
                idx = j + num_cols * i
                if idx >= num:
                    break
                im = ax[i, j].imshow(layer[0][idx], interpolation='nearest',
                                     clim=limits)
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)
    unit = ' [kHz]' if title == 'Spikerates' else ''
    f.suptitle('{} \n of layer {}'.format(title + unit, layer[1]), fontsize=20)
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


def plot_activations(model, x_test, path):
    """Plot activations of a network.

    Parameters
    ----------

    model: keras.models.Model
        Keras model.

    x_test: ndarray
        The samples.

    path: str
        Where to save plot.
    """

    from snntoolbox.conversion.utils import get_activations_batch
    from snntoolbox.simulation.utils import get_sample_activity_from_batch

    activations_batch = get_activations_batch(model, x_test)
    activations = get_sample_activity_from_batch(activations_batch, 0)
    for i in range(len(activations)):
        label = activations[i][1]
        print("Plotting layer {}".format(label))
        if not os.path.exists(path):
            os.makedirs(path)
        j = str(i) if i > 9 else '0' + str(i)
        plot_layer_activity(activations[i], j+label, path)


def plot_activations_minus_rates(activations, rates, label, path=None):
    """Plot spikerates minus activations for a specific layer.

    Spikerates and activations are each normalized before subtraction.
    The neurons in the layer are arranged in a line or on a 2D grid, depending
    on layer type.

    Activity is encoded by color.

    Parameters
    ----------

    activations: ndarray
        The activations of a layer. The shape is that of the original layer,
        e.g. (32, 28, 28) for 32 feature maps of size 28x28.
    rates: ndarray
        The spikerates of a layer. The shape is that of the original layer,
        e.g. (32, 28, 28) for 32 feature maps of size 28x28.
    label: str
        Layer label.
    path: Optional[str]
        If not ``None``, specifies where to save the resulting image. Else,
        display plots without saving.
    """

    activations_norm = activations / np.max(activations)
    rates_norm = rates / np.max(rates) if np.max(rates) != 0 else rates
    plot_layer_activity(
        (activations_norm - rates_norm, label),
        str('Activations_minus_Spikerates'), path, limits=(-1, 1))


def plot_layer_correlation(rates, activations, title, config, path=None):
    """
    Plot correlation between spikerates and activations of a specific layer,
    as 2D-dot-plot.

    Parameters
    ----------

    rates: np.array
        The spikerates of a layer, flattened to 1D.
    activations: Union[ndarray, Iterable]
        The activations of a layer, flattened to 1D.
    title: str
        Plot title.
    config: configparser.ConfigParser
        Settings.
    path: Optional[str]
        If not ``None``, specifies where to save the resulting image. Else,
        display plots without saving.
    """

    # Determine percentage of saturated neurons. Need to subtract one time step
    dt = config.getfloat('simulation', 'dt')
    duration = config.getint('simulation', 'duration')
    p = np.mean(np.greater_equal(rates, 1000 / dt - 1000 / duration / dt))

    plt.figure()
    plt.plot(activations, rates, '.')
    plt.annotate("{:.2%} units saturated.".format(p), xy=(1, 1),
                 xycoords='axes fraction', xytext=(-200, -20),
                 textcoords='offset points')
    plt.title(title, fontsize=20)
    plt.locator_params(nbins=4)
    lim = max([1.1, max(activations)])
    plt.xlim([0, lim])
    plt.ylim([0, lim])
    plt.xlabel('ANN activations', fontsize=16)
    plt.ylabel('SNN spikerates [Hz]', fontsize=16)
    if path is not None:
        filename = '5Correlation'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_correlations(spikerates, layer_activations):
    """Plot the correlation between SNN spiketrains and ANN activations.

    For each layer, the method draws a scatter plot, showing the correlation
    between the average firing rate of neurons in the SNN layer and the
    activation of the corresponding neurons in the ANN layer.

    Parameters
    ----------

    spikerates: list of tuples ``(spikerate, label)``.

        ``spikerate`` is a 1D array containing the mean firing rates of the
        neurons in a specific layer.

        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'3Dense'``.

    layer_activations: list of tuples ``(activations, label)``
        Each entry represents a layer in the ANN for which an activation can be
        calculated (e.g. ``Dense``, ``Conv2D``).

        ``activations`` is an array of the same dimension as the corresponding
        layer, containing the activations of Dense or Convolution layers.

        ``label`` is a string specifying the layer type, e.g. ``'Dense'``.
    """

    num_layers = len(layer_activations)
    # Determine optimal shape for rectangular arrangement of plots
    num_rows = int(np.ceil(np.sqrt(num_layers)))
    num_cols = int(np.ceil(num_layers / num_rows))
    f, ax = plt.subplots(num_rows, num_cols, squeeze=False,
                         figsize=(8, 1 + num_rows * 4))
    for i in range(num_rows):
        for j in range(num_cols):
            layer_num = j + i * num_cols
            if layer_num >= num_layers:
                break
            ax[i, j].plot(layer_activations[layer_num][0].flatten(),
                          spikerates[layer_num][0], '.')
            ax[i, j].set_title(spikerates[layer_num][1], fontsize='medium')
            ax[i, j].locator_params(nbins=4)
            ax[i, j].set_xlim([None,
                               np.max(layer_activations[layer_num][0]) * 1.1])
            ax[i, j].set_ylim([None, max(spikerates[layer_num][0]) * 1.1])
    f.suptitle('ANN-SNN correlations', fontsize=20)
    f.subplots_adjust(wspace=0.3, hspace=0.3)
    f.text(0.5, 0.04, 'SNN spikerates (Hz)', ha='center', fontsize=16)
    f.text(0.04, 0.5, 'ANN activations', va='center', rotation='vertical',
           fontsize=16)


def get_pearson_coefficients(spikerates_batch, activations_batch, max_rate):
    """
    Compute Pearson coefficients.

    Parameters
    ----------

    spikerates_batch : 
    activations_batch :
    max_rate: float
        Highest spike rate.

    Returns
    -------
    
    co: list

    """

    co = []
    for layer_num in range(len(spikerates_batch)):
        c = []
        for sample in range(len(spikerates_batch[0][0])):
            s = spikerates_batch[layer_num][0][sample].flatten()
            a = activations_batch[layer_num][0][sample].flatten()
            if layer_num < len(spikerates_batch) - 1:
                # Remove points at origin and saturated units, except for
                # output layer (has too few units and gets activation of 1
                # because of softmax).
                ss = []
                aa = []
                for sss, aaa in zip(s, a):
                    if (sss > 0 or aaa > 0) and aaa < max_rate:
                        ss.append(sss)
                        aa.append(aaa)
                s = ss
                a = aa
            c.append(np.corrcoef(s, a)[0, 1])
        co.append(c)

    return co


def plot_pearson_coefficients(spikerates_batch, activations_batch, config,
                              path=None):
    """
    Plot the Pearson correlation coefficients for each layer, averaged over one
    mini batch.

    Parameters
    ----------

    spikerates_batch: list[tuple[np.array, str]]
        Each entry in ``spikerates_batch`` contains a tuple
        ``(spikerates, label)`` for each layer of the network (for the first
        batch only, and excluding ``Flatten`` layers).

        ``spikerates`` contains the average firing rates of all neurons in a
        layer. It has the same shape as the original layer, e.g.
        (batch_size, n_features, n_rows, n_cols) for a convolution layer.

        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'03Dense'``.

    activations_batch: list[tuple[np.array, str]]
        Contains the activations of a net. Same structure as
        ``spikerates_batch``.

    config: configparser.ConfigParser
        Settings.

    path: Optional[str]
        Where to save the output.
    """

    from snntoolbox.utils.utils import extract_label

    max_rate = 1. / config.getfloat('simulation', 'dt')
    co = get_pearson_coefficients(spikerates_batch, activations_batch, max_rate)

    # Average over batch
    corr = np.mean(co, axis=1)
    std = np.std(co, axis=1)

    labels = [sp[1] for sp in spikerates_batch]
    if config.getboolean('output', 'use_simple_labels'):
        labels = [extract_label(label)[1] for label in labels]

    plt.figure()
    plt.bar([i + 0.1 for i in range(len(corr))], corr, width=0.8, yerr=std,
            color=(0.8, 0.8, 0.8))
    plt.ylim([0, 1])
    plt.xlim([0, len(corr)])
    plt.xticks([i + 0.5 for i in range(len(labels))], labels, rotation=90)
    plt.tick_params(bottom='off')
    plt.title('Correlation between ANN activations \n and SNN spikerates,\n ' +
              'averaged over {} samples'.format(len(spikerates_batch[0][0])))
    plt.ylabel('Pearson Correlation Coefficient')
    if path is not None:
        filename = 'Pearson'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_hist(h, title=None, layer_label=None, path=None, scale_fac=None):
    """Plot a histogram over two datasets.

    Parameters
    ----------

    h: dict
        Dictionary of datasets to plot in histogram.
    title: string, optional
        Title of histogram.
    layer_label: string, optional
        Label of layer from which data was taken.
    path: string, optional
        If not ``None``, specifies where to save the resulting image. Else,
        display plots without saving.
    scale_fac: float, optional
        The value with which parameters are normalized (maximum of activations
        or parameter value of a layer). If given, will be insterted into plot
        title.
    """

    keys = sorted(h.keys())
    plt.hist([h[key] for key in keys], label=keys, log=True, bottom=1,
             bins=100, histtype='stepfilled', alpha=0.5)
    if scale_fac:
        plt.axvline(scale_fac, color='red', linestyle='dashed', linewidth=2,
                    label='scale factor')
    plt.legend()
    plt.locator_params(axis='x', nbins=10)
    if title and layer_label:
        if 'Spikerates' in title:
            filename = '4' + title + '_distribution'
            unit = '[Hz]'
        else:
            filename = layer_label + '_' + title + '_distribution'
            unit = ''
        facs = "Applied divisor: {:.2f}".format(scale_fac) if scale_fac else ''
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


def plot_activ_hist(h, title=None, layer_label=None, path=None, scale_fac=None):
    """Plot a histogram over all activities of a network.

    Parameters
    ----------

    h: dict
        Dictionary of datasets to plot in histogram.
    title: string, optional
        Title of histogram.
    layer_label: string, optional
        Label of layer from which data was taken.
    path: string, optional
        If not ``None``, specifies where to save the resulting image. Else,
        display plots without saving.
    scale_fac: float, optional
        The value with which parameters are normalized (maximum of activations
        or parameter value of a layer). If given, will be insterted into plot
        title.
    """

    keys = sorted(h.keys())
    plt.hist([h[key] for key in keys], label=keys, bins=1000, edgecolor='blue',
             histtype='stepfilled', log=True, bottom=1)
    plt.xlabel('ANN activations')
    plt.ylabel('Count')
    plt.xlim(xmin=0)
    if scale_fac:
        plt.axvline(scale_fac, color='red', linestyle='dashed', linewidth=2,
                    label='scale factor')
    plt.legend()
    plt.locator_params(axis='x', nbins=10)
    if title and layer_label:
        filename = layer_label + '_' + 'activ_distribution'
        facs = "Applied divisor: {:.2f}".format(scale_fac) if scale_fac else ''
        plt.title('{} distribution \n of layer {} \n {}'.format(
            title, layer_label, facs))
    else:
        plt.title('Distribution')
        filename = 'Activity_distribution'
    if path:
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_max_activ_hist(h, title=None, layer_label=None, path=None,
                        scale_fac=None):
    """Plot a histogram over the maximum activations.

    Parameters
    ----------

    h: dict
        Dictionary of datasets to plot in histogram.
    title: string, optional
        Title of histogram.
    layer_label: string, optional
        Label of layer from which data was taken.
    path: string, optional
        If not ``None``, specifies where to save the resulting image. Else,
        display plots without saving.
    scale_fac: float, optional
        The value with which parameters are normalized (maximum of activations
        or parameter value of a layer). If given, will be insterted into plot
        title.
    """

    keys = sorted(h.keys())
    plt.hist([h[key] for key in keys], label=keys, bins=1000, edgecolor='blue',
             histtype='stepfilled')
    plt.xlabel('Maximum ANN activations')
    plt.ylabel('Sample count')
    if scale_fac:
        plt.axvline(scale_fac, color='red', linestyle='dashed', linewidth=2,
                    label='scale factor')
    plt.legend()
    plt.locator_params(axis='x', nbins=10)
    if title and layer_label:
        filename = layer_label + '_' + 'maximum_activity_distribution'
        facs = "Applied divisor: {:.2f}".format(scale_fac) if scale_fac else ''
        plt.title('{} distribution \n of layer {} \n {}'.format(
            title, layer_label, facs))
    else:
        plt.title('Distribution')
        filename = 'Maximum_activity_distribution'
    if path:
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_hist_combined(data, path=None):
    """Plot a histogram over several datasets.

    Parameters
    ----------

    data: dict
        Dictionary of datasets to plot in histogram.
    path: string, optional
        If not ``None``, specifies where to save the resulting image. Else,
        display plots without saving.
    """

    # Flatten containers in data, because we don't need the original
    # 3D-structure of a layer in a histogram.
    h = {}
    for (key, val) in data.items():
        l = []
        for a in val:
            l += list(a[0].flatten())
        h.update({key: l})

    keys = sorted(h.keys())
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', which='both')
    ax.get_xaxis().set_visible(False)
    axes = [ax]
    fig.subplots_adjust(top=0.8)
    # noinspection PyUnresolvedReferences
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


def plot_param_sweep(results, n, params, param_name, param_logscale):
    """Plot accuracy versus parameter.

    Parameters
    ----------

    results: list[float]
        The accuracy or loss for a number of experiments, each of which used
        different parameters.
    n: int
        The number of test samples used for each experiment.
    params: list[float]
        The parameter values that changed during each experiment.
    param_name: str
        The name of the parameter that varied.
    param_logscale: bool
        Whether to plot the parameter axis in log-scale.
    """

    from snntoolbox.utils.utils import wilson_score

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


def plot_spiketrains(layer, path=None):
    """Plot which neuron fired at what time during the simulation.

    Parameters
    ----------

    layer: tuple[np.array, str]
        ``(spiketimes, label)``.

        ``spiketimes`` is a 2D array where the first index runs over the number
        of neurons in the layer, and the second index contains the spike times
        of the specific neuron.

        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'3Dense'``.

    path: Optional[str]
        If not ``None``, specifies where to save the resulting image. Else,
        display plots without saving.
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
    plt.title('Spiketrains \n of layer {}'.format(layer[1]))
    plt.xlabel('time [ms]')
    plt.ylabel('neuron index')
    plt.xlim(1, shape[-1]+1)
    if path is not None:
        filename = '7Spiketrains'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_potential(times, layer, config, show_legend=False, path=None):
    """Plot the membrane potential of a layer.

    Parameters
    ----------

    times: np.array
        The time values where the potential was sampled.

    layer: tuple[np.array, str]
        ``(vmem, label)``.

        ``vmem`` is a 2D array where the first index runs over the number of
        neurons in the layer, and the second index contains the membrane
        potential of the specific neuron.

        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'3Dense'``.

    config: configparser.ConfigParser
        Settings.

    show_legend: bool
        If ``True``, shows the legend indicating the neuron indices and lines
        like ``v_thresh``, ``v_rest``, ``v_reset``. Recommended only for layers
        with few neurons.

    path: Optional[str]
        If not ``None``, specifies where to save the resulting image. Else,
        display plots without saving.
    """

    v_thresh = config.getfloat('cell', 'v_thresh')
    v_reset = config.getfloat('cell', 'v_reset')

    plt.figure()
    # Transpose layer array to get slices of vmem values for each neuron.
    layer_flat = np.reshape(layer[0], (-1, layer[0].shape[-1]))
    for (neuron, vmem) in enumerate(layer_flat):
        plt.plot(times, vmem)
    plt.plot(times, np.ones_like(times) * v_thresh, 'r--', label='V_thresh')
    plt.plot(times, np.ones_like(times) * v_reset, 'b-.', label='V_reset')
    plt.ylim([v_reset - 0.1, v_thresh + 0.1])
    if show_legend:
        plt.legend(loc='upper left', prop={'size': 15})
    plt.xlabel('Time [ms]')
    plt.ylabel('Membrane potential')
    plt.title('Membrane potential for neurons \n in layer {}'.format(layer[1]))
    if path is not None:
        filename = 'Potential'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(y_test, y_pred, path=None, class_labels=None):
    """

    Parameters
    ----------

    y_test: list
    y_pred: list
    path: Optional[str]
        Where to save the output.
    class_labels: Optional[list]
        List of class labels.
    """

    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("ERROR: Failed to plot confusion matrix: sklearn package not "
              "installed. Do 'pip install sklearn' to install.\n")
        confusion_matrix = None

    if confusion_matrix is None:
        return

    cm = confusion_matrix(y_test, y_pred, class_labels)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    if class_labels:
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels)
        plt.yticks(tick_marks, class_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if path is not None:
        filename = 'Confusion'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_error_vs_time(top1err_d_t, top5err_d_t, duration, dt,
                       top1err_ann=None, top5err_ann=None, path=None):
    """Plot classification error over time.

    Parameters
    ----------

    top1err_d_t: np.array
        Batch of top-1 errors over time. Shape: (num_samples, duration).
        Data type: boolean (correct / incorrect classification).
    top5err_d_t: np.array
        Batch of top-5 errors over time. Shape: (num_samples, duration).
        Data type: boolean (correct / incorrect classification).
    duration: int
        Simulation duration.
    dt: float
        Simulation time resolution.
    top1err_ann: Optional[float]
        The top-1 error of the ANN.
    top5err_ann: Optional[float]
        The top-5 error of the ANN.
    path: Optional[str]
        Where to save the output.
    """

    top1err_t = np.mean(top1err_d_t, 0) * 100
    top1std_t = np.std(top1err_d_t, 0) * 100
    top5err_t = np.mean(top5err_d_t, 0) * 100
    top5std_t = np.std(top5err_d_t, 0) * 100

    plt.figure()
#    plt.title('Error vs simulation time')
    time = np.arange(0, duration, dt)
    plt.plot(time, top1err_t, '.g', label='SNN top-1')
    plt.plot(time, top5err_t, 'xb', label='SNN top-5')
    plt.fill_between(time, top1err_t-top1std_t, top1err_t+top1std_t, alpha=0.1,
                     color='green')
    plt.fill_between(time, top5err_t-top5std_t, top5err_t+top5std_t, alpha=0.1,
                     color='blue')
    if top1err_ann:
        plt.hlines(top1err_ann*100, 0, time[-1], label='ANN top-1',
                   colors='red', linestyle='-.')
    if top5err_ann:
        plt.hlines(top5err_ann*100, 0, time[-1], label='ANN top-5',
                   colors='orange', linestyle='--')
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel('Error [%]')
    plt.xlabel('Simulation time [ms] in steps of {} ms.'.format(dt))
    if path is not None:
        filename = 'Error_vs_time'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_ops_vs_time(operations_b_t, duration, dt, path=None):
    """Plot total number of operations over time.

    Parameters
    ----------

    operations_b_t : ndarray
        Number of operations. Shape: (batch_size, num_timesteps)
    duration: int
        Simulation duration.
    dt: float
        Simulation time resolution.
    path: Optional[str]
        Where to save the output.
    """

    plt.figure()
    plt.title('SNN operations')
    time = np.arange(0, duration, dt)
    mean_ops_t = np.mean(operations_b_t, 0)
    std_ops_t = np.std(operations_b_t, 0)
    plt.plot(time, mean_ops_t, '.b')
    plt.fill_between(time, mean_ops_t-std_ops_t, mean_ops_t+std_ops_t,
                     alpha=0.1, color='b')
    plt.ylim(0, None)
    plt.ylabel('MOps')
    plt.xlabel('Simulation time [ms] in steps of {} ms.'.format(dt))
    if path is not None:
        filename = 'operations_t'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_spikecount_vs_time(spiketrains_n_b_l_t, duration, dt, path=None):
    """Plot total spikenumber over time.

    Parameters
    ----------

    spiketrains_n_b_l_t:
    duration: int
        Simulation duration.
    dt: float
        Simulation time resolution.
    path: Optional[str]
        Where to save the output.
    """

    # batch time dimensions
    b_t_shape = (spiketrains_n_b_l_t[0][0].shape[0],
                 spiketrains_n_b_l_t[0][0].shape[-1])
    spikecounts_b_t = np.zeros(b_t_shape)
    for n in range(len(spiketrains_n_b_l_t)):  # Loop over layers
        spiketrains_b_l_t = np.greater(spiketrains_n_b_l_t[n][0], 0)
        reduction_axes = tuple(np.arange(1, spiketrains_b_l_t.ndim-1))
        spikecounts_b_t += np.sum(spiketrains_b_l_t, reduction_axes)
    cum_spikecounts_b_t = np.cumsum(spikecounts_b_t, 1)

    plt.figure()
    plt.title('SNN spike count')
    time = np.arange(0, duration, dt)
    cum_spikecounts_t = np.mean(cum_spikecounts_b_t, 0)
    std_t = np.std(cum_spikecounts_b_t, 0)
    plt.plot(time, cum_spikecounts_t, '.b')
    plt.fill_between(time, cum_spikecounts_t-std_t, cum_spikecounts_t+std_t,
                     alpha=0.1, color='b')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.ylim(0, None)
    plt.ylabel('# spikes')
    plt.xlabel('Simulation time [ms] in steps of {} ms.'.format(dt))
    if path is not None:
        filename = 'Total_spike_count'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_input_image(x, label, path=None):
    """Show an input image.

    Parameters
    ----------
    x: ndarray
        The sample to plot.
    label: int
        Class label (index) of sample.
    path: Optional[str]
        Where to save the image.
    """

    plt.figure()
    plt.title('Input image (class: {})'.format(label))
    if x.ndim == 1:
        try:
            x = np.reshape(x, (1, int(np.sqrt(len(x))), -1))
        except RuntimeError:
            return
    x = np.transpose(x, (1, 2, 0)) if x.shape[0] == 3 else x[0]
    plt.imshow(x)
    if path is not None:
        filename = 'input_image'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_history(h):
    """Plot the training and validation loss and accuracy at each epoch.

    Parameters
    ----------

    h: Keras history object
        Contains the training and validation loss and accuracy at each epoch
        during training.
    """

    plt.figure()

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
