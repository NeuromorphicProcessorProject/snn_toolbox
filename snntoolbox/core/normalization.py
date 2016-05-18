# -*- coding: utf-8 -*-
"""
Functions to normalize parameters of a network.

Created on Mon Mar  7 17:13:18 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import os
import numpy as np
from snntoolbox import echo
from snntoolbox.config import globalparams
from snntoolbox.io.save import save_model
from snntoolbox.io.plotting import plot_hist
import theano

standard_library.install_aliases()


def normalize_weights(model, X_train, filename=None):
    """
    Normalize the weights of a network.

    The weights of each layer are normalized with respect to the maximum
    activation.

    Parameters
    ----------

    model : network object
        A network object of the ``model_lib`` language, e.g. keras.
    X_train : float32 array
        The input samples to use for determining the layer activations.
        With data of the form (channels, num_rows, num_cols),
        X_test has dimension (1, channels*num_rows*num_cols) for a multi-layer
        perceptron, and (1, channels, num_rows, num_cols) for a convnet.
    filename : string, optional
        Name of the file where the normalized model should be stored. Will be
        appended with ``_normWeights``. If the file already exists, the program
        will ask for confirmation to overwrite. If no filename is given,
        the modified net will not be written to disk. If the network is
        converted to a spiking network after normalization, the converted net
        will be stored to disk with normalized weights. The filename of this
        SNN will not be changed (no ``_normWeights`` appended), because the
        spiking net is assumed to use normalized weights by default.

    Returns
    -------

    model : network object
        A network object of the ``model_lib`` language, e.g. keras.
    """

    def get_activations(get_activ, X_train):
        shape = list(get_activ(X_train[:1], 0).shape)
        shape[0] = X_train.shape[0]
        activations = np.empty(shape)
        num_batches = int(np.ceil(X_train.shape[0]/globalparams['batch_size']))
        for batch_idx in range(num_batches):
            # Determine batch indices.
            max_idx = min((batch_idx + 1) * globalparams['batch_size'],
                          X_train.shape[0])
            batch_idxs = range(batch_idx * globalparams['batch_size'], max_idx)
            batch = X_train[batch_idxs, :]
            activations[batch_idxs] = get_activ(batch, 0)
        return activations

    def norm_weights(weights, activations, previous_fac):
        # Skip last batch if batch_size was chosen such that the dataset could
        # not be divided into an integer number of equal-sized batches.
        end = -1 if len(activations[0]) != len(activations[-1]) else None
        activation_max = np.percentile(activations[:end], 95)
        weight_max = np.max(weights[0])  # Disregard biases
        scale_fac = np.max([weight_max, activation_max])
        print("Done. \n Maximum value: {:.2f}.".format(scale_fac))
        # Normalization factor is the ratio of the max values of the
        # previous to this layer.
        applied_fac = scale_fac / previous_fac
        print("Applied divisor: {:.2f}.".format(applied_fac))
        return [x / applied_fac for x in weights], scale_fac, applied_fac

    echo("Normalizing weights:\n")
    newpath = os.path.join(globalparams['log_dir_of_current_run'],
                           'normalization')
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    previous_fac = 1
    m = model['model']
    if globalparams['model_lib'] == 'keras':
        from keras import backend as K
        # Loop through all layers, looking for activation layers
        for idx, layer in enumerate(m.layers):
            if layer.__class__.__name__ == 'Activation':
                label = m.layers[idx-1].get_config()['name']
                print("Calculating output of activation layer {}".format(idx) +
                      " following layer {} with shape {}...".format(
                      label, layer.output_shape))
                get_activ = theano.function([m.layers[0].input,
                                             K.learning_phase()],
                                            layer.output,
                                            allow_input_downcast=True,
                                            on_unused_input='ignore')
                weights = m.layers[idx-1].get_weights()
                activations = get_activations(get_activ, X_train)
                weights_norm, previous_fac, applied_fac = norm_weights(
                    weights, activations, previous_fac)
                m.layers[idx-1].set_weights(weights_norm)
                activations_norm = get_activations(get_activ, X_train)
                # For memory reasons, use only a fraction of samples for
                # plotting a histogram of activations.
                frac = int(len(activations) / 5000)
                print(frac)
                activation_dict = {'Activations': activations[:frac].flatten(),
                                   'Activations_norm':
                                       activations_norm[:frac].flatten()}
                weight_dict = {'Weights': weights[0].flatten(),
                               'Weights_norm': weights_norm[0].flatten()}
                layer_label = '0' + str(idx-1) + label if idx < 10 else \
                    str(idx-1) + label
                plot_hist(activation_dict, 'Activation', layer_label, newpath,
                          previous_fac, applied_fac)
                plot_hist(weight_dict, 'Weight', layer_label, newpath)
    elif globalparams['model_lib'] == 'lasagne':
        import lasagne
        from snntoolbox.config import activation_layers
        layers = lasagne.layers.get_all_layers(m)
        # Loop through all layers, looking for activation layers
        for idx, layer in enumerate(layers):
            label = layer.__class__.__name__
            if label in activation_layers:
                print("Calculating output of layer {} with shape {}...".format(
                      label, layer.output_shape))
                get_activ = theano.function(
                    [layers[0].input_var, theano.tensor.scalar()],
                    lasagne.layers.get_output(layer, layers[0].input_var),
                    allow_input_downcast=True)
                weights = layer.W.get_value()
                activations = get_activations(get_activ, X_train)
                weights_norm, previous_fac = norm_weights(weights, activations,
                                                          previous_fac)
                layer.W.set_value(weights_norm)

    # Write out weights
    if filename is not None:
        save_model(m, filename=filename+'_normWeights')
    model['model'] = m
    return model
