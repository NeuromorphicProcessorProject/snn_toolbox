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
from snntoolbox.io.plotting import plot_activity_distribution_layer
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
        num_batches = int(np.ceil(X_train.shape[0]/globalparams['batch_size']))
        activations = []
        for batch_idx in range(num_batches):
            # Determine batch indices.
            max_idx = min((batch_idx + 1) * globalparams['batch_size'],
                          X_train.shape[0])
            batch_idxs = range(batch_idx * globalparams['batch_size'], max_idx)
            batch = X_train[batch_idxs, :]
            activations.append(get_activ(batch, 0))
        return activations

    def norm_weights(weights, activations, norm_fac):
        activation_max = np.max(np.mean(activations, axis=0))
        weight_max = np.max(weights[0])  # Disregard biases
        total_max = np.max([weight_max, activation_max])
        print("Done. Maximum value: {:.2f}.".format(total_max))
        # Normalization factor is the ratio of the max values of the
        # previous to this layer.
        norm_fac /= total_max
#        norm_fac = 1 / total_max
        return [x * norm_fac for x in weights], norm_fac

    echo("Normalizing weights:\n")
    newpath = os.path.join(globalparams['path'], 'log', 'gui', 'normalization')
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    norm_fac = 1
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
                weights_norm, norm_fac = norm_weights(weights, activations,
                                                      norm_fac)
                m.layers[idx-1].set_weights(weights_norm)
                activation_dict = {'Activations': activations[0],
                                   'Weights': weights[0],
                                   'weights_norm': weights_norm[0]}
                title = '0' + str(idx) + label if idx < 10 else str(idx)+label
                plot_activity_distribution_layer(activation_dict, title,
                                                 newpath)
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
                weights_norm, norm_fac = norm_weights(weights, activations,
                                                      norm_fac)
                layer.W.set_value(weights_norm)

    # Write out weights
    if filename is not None:
        save_model(m, filename=filename+'_normWeights')
    model['model'] = m
    return model
