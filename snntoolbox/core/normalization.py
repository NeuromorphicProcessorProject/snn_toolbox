# -*- coding: utf-8 -*-
"""
Functions to normalize parameters of a network.

Citation
--------

``Diehl, P.U. and Neil, D. and Binas, J. and Cook, M. and Liu, S.C. and
Pfeiffer, M. Fast-Classifying, High-Accuracy Spiking Deep Networks Through
Weight and Threshold Balancing, IEEE International Joint Conference on Neural
Networks (IJCNN), 2015``

Created on Mon Mar  7 17:13:18 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import numpy as np
from snntoolbox import echo
from snntoolbox.config import globalparams
from snntoolbox.io.save import save_model
# Turn off "Warning: The downsample module has been moved to the pool module."
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    warnings.warn('deprecated', UserWarning)
    import theano

standard_library.install_aliases()


def normalize_weights(model, X_train, path, filename=None):
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
    path : string
        Location of the ANN model to load.
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

    def norm_weights(weights, get_activ, X_train, norm_fac):
        activation_max = np.mean([np.max(get_activ(
            np.array(X, ndmin=X_train.ndim))) for X in X_train])
        weight_max = np.max(weights[0])  # Disregard biases
        total_max = np.max([weight_max, activation_max])
        print("Done. Maximum value: {:.2f}.".format(total_max))
        # Normalization factor is the ratio of the max values of the
        # previous to this layer.
        norm_fac /= total_max
#        norm_fac = 1 / total_max
        return [x * norm_fac for x in weights], norm_fac

    echo("Normalizing weights:\n")
    norm_fac = 1
    m = model['model']
    if globalparams['model_lib'] == 'keras':
        # Loop through all layers, looking for activation layers
        for idx, layer in enumerate(m.layers):
            if layer.__class__.__name__ == 'Activation':
                print("Calculating output of activation layer {}".format(idx) +
                      " following layer {} with shape {}...".format(
                      m.layers[idx-1].get_config()['name'],
                      layer.output_shape))
                get_activ = theano.function([m.layers[0].input],
                                            layer.get_output(train=False),
                                            allow_input_downcast=True)
                weights = m.layers[idx-1].get_weights()
                weights_norm, norm_fac = norm_weights(weights, get_activ,
                                                      X_train, norm_fac)
                m.layers[idx-1].set_weights(weights_norm)
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
                    [layers[0].input_var],
                    lasagne.layers.get_output(layer, layers[0].input_var),
                    allow_input_downcast=True)
                weights = layer.W.get_value()
                weights_norm, norm_fac = norm_weights(weights, get_activ,
                                                      X_train, norm_fac)
                layer.W.set_value(weights_norm)

    # Write out weights
    if filename is not None:
        save_model(m, filename=filename+'_normWeights')
    model['model'] = m
    return model
