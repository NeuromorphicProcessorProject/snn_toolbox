# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:03:06 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import os

from snntoolbox.config import settings

standard_library.install_aliases()


class SNN():
    """
    Represent a neural network.

    Implements a class that contains essential information about the
    architecture and weights of a neural network, while being independent of
    the library from which the original network was built.

    The constructor ``__init__()`` performs the extraction of the
    attributes of interest.

    Parameters
    ----------

        model : network object
            A network object of the ``model_lib`` language, e.g. keras.

    Attributes
    ----------

        - weights : array
            Weights connecting the input layer.

        - biases : array
            Biases of the network. For conversion to spiking nets, zero biases
            are found to work best.

        - input_shape : list
            The dimensions of the input sample.

        - layers : list
            List of all the layers of the network, where each layer contains a
            dictionary with keys

            - layer_num : int
                Index of layer.

            - layer_type : string
                Describing the type, e.g. `Dense`, `Convolution`, `Pool`.

            - output_shape : list
                The output dimensions of the layer.

            In addition, `Dense` and `Convolution` layer types contain

            - weights : array
                The weight parameters connecting the layer with the next.

            `Convolution` layers contain further

            - nb_col : int
                The x-dimension of filters.

            - nb_row : int
                The y-dimension of filters.

            - border_mode : string
                How to handle borders during convolution, e.g. `full`, `valid`,
                `same`.

            `Pooling` layers contain

            - pool_size : list
                Specifies the subsampling factor in each dimension.

            - strides : list
                The stepsize in each dimension during pooling.

    The initializer is meant to be extended by functionality to extract any
    model written in any of the common neural network libraries, e.g. keras,
    theano, caffe, torch, etc.

    The simplified returned network can then be used in the SNN conversion and
    simulation toolbox.

    """

    def __init__(self, model):
        self.import_modules()
        self.model = model['model']
        ann = self.model_lib.extract(model['model'])
        self.compiled_snn = self.target_sim.SNN_compiled(ann)
        self.input_shape = ann['input_shape']
        self.layers = ann['layers']
        self.labels = ann['labels']
        self.layer_idx_map = ann['layer_idx_map']
        self.val_fn = model['val_fn']

    def import_modules(self):
        from importlib import import_module
        self.model_lib = import_module('snntoolbox.model_libs.' +
                                       settings['model_lib'] + '_input_lib')
        self.target_sim = import_module('snntoolbox.target_simulators.' +
                                        settings['simulator'] + '_target_sim')

    def evaluate_ann(self, X_test, Y_test, **kwargs):
        """
        Evaluate the performance of a network.

        Wrapper for the evaluation functions of specific input neural network
        libraries ``globalparams['model_lib']`` like keras, caffe, torch, etc.

        Needs to be extended further: Supports only keras so far.

        Parameters
        ----------

        ann : network object
            The neural network of ``backend`` type.
        X_test : float32 array
            The input samples to test.
            With data of the form (channels, num_rows, num_cols),
            X_test has dimension (num_samples, channels*num_rows*num_cols)
            for a multi-layer perceptron, and
            (num_samples, channels, num_rows, num_cols) for a convolutional
            net.
        Y_test : float32 array
            Ground truth of test data. Has dimesion (num_samples, num_classes).

        Returns
        -------

        The output of the ``model_lib`` specific evaluation function, e.g. the
        score of a keras model.

        """

        score = self.model_lib.evaluate(self.val_fn, X_test, Y_test)

        print("Test score: {:.2f}\n".format(score[0]))
        print("Test accuracy: {:.2%}\n".format(score[1]))

        return score

    def normalize_weights(self, X_train):
        """
        Normalize the weights of a network.

        The weights of each layer are normalized with respect to the maximum
        activation.

        Parameters
        ----------

        X_train : float32 array
            The input samples to use for determining the layer activations.
            With data of the form (channels, num_rows, num_cols),
            X_test has dimension (1, channels*num_rows*num_cols) for a
            multi-layer perceptron, and (1, channels, num_rows, num_cols) for a
            convnet.

        """

        from snntoolbox.io_utils.plotting import plot_hist
        from snntoolbox.core.util import get_activations_layer, norm_weights

        print("Normalizing weights:\n")
        newpath = os.path.join(settings['log_dir_of_current_run'],
                               'normalization')
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        previous_fac = 1
        # Loop through all layers, looking for activation layers
        for idx, layer in enumerate(self.layers):
            # Skip layer if not preceeded by a layer with weights
            if idx == 0 or 'weights' not in self.layers[idx-1].keys():
                # A pooling layer has no weights but we can calculate
                # activations. Compile function based on normalized lower
                # layers:
                if 'get_activ' in layer.keys():
                    layer.update(
                        {'get_activ_norm':
                         self.model_lib.get_activ_fn_for_layer(
                             self.model, self.layer_idx_map[idx])})
                continue
            label = self.labels[idx-1]
            print("Calculating output of activation layer {}".format(idx) +
                  " following layer {} with shape {}...".format(
                  label, layer['output_shape']))
            weights = self.layers[idx-1]['weights']
            activations = get_activations_layer(layer['get_activ'], X_train)
            weights_norm, previous_fac, applied_fac = norm_weights(
                weights, activations, previous_fac)
            # Update model with modified weights
            self.set_layer_params(weights_norm, idx-1)
            # Compile new theano function with modified weights
            get_activ_norm = self.model_lib.get_activ_fn_for_layer(
                self.model, self.layer_idx_map[idx])
            layer.update({'get_activ_norm': get_activ_norm})
            activations_norm = get_activations_layer(get_activ_norm, X_train)
            # For memory reasons, use only a fraction of samples for
            # plotting a histogram of activations.
            frac = 10  # int(len(activations) / 100)
            activation_dict = {'Activations': activations[:frac].flatten(),
                               'Activations_norm':
                                   activations_norm[:frac].flatten()}
            weight_dict = {'Weights': weights[0].flatten(),
                           'Weights_norm': weights_norm[0].flatten()}
            plot_hist(activation_dict, 'Activation', label, newpath,
                      previous_fac, applied_fac)
            plot_hist(weight_dict, 'Weight', label, newpath)

    def set_layer_params(self, params, i):
        self.layers[i]['weights'] = params
        self.model_lib.set_layer_params(self.model, params,
                                        self.layer_idx_map[i])

    def get_params(self):
        return [l['weights'] for l in self.layers if 'weights' in l.keys()]

    def save(self, path=None, filename=None):
        """
        Write model architecture and weights to disk.

        Parameters
        ----------

        """

        from snntoolbox.io_utils.save import confirm_overwrite

        if not path:
            path = settings['path']
        if not filename:
            filename = settings['filename']

        print("Saving model to {}...".format(path))
        # Create directory if not existent yet.
        if not os.path.exists(path):
            os.makedirs(path)

        filepath = os.path.join(path, filename)
        if confirm_overwrite(filepath + '.h5'):
            self.save_weights(filepath + '.h5')

        if confirm_overwrite(filepath + '.json'):
            self.save_config(filepath + '.json')

        print("Done.\n")

    def get_config(self):
        skip_keys = ['weights', 'get_activ', 'get_activ_norm', 'model']
        layer_config = []
        for layer in self.layers:
            layer_config.append([{key: layer[key]} for key in layer.keys()
                                 if key not in skip_keys])
        return {'name': self.__class__.__name__,
                'input_shape': self.input_shape,
                'layers': layer_config}

    def save_config(self, path):
        from snntoolbox.io_utils.save import to_json
        to_json(self.get_config(), path)

    def save_weights(self, path):
        """
        Dump all layer weights to a HDF5 file.
        """

        import h5py

        f = h5py.File(path, 'w')

        f.attrs['layer_names'] = [l.encode('utf8') for l in self.labels]

        for layer in self.layers:
            if 'weights' not in layer.keys():
                continue
            g = f.create_group(layer['label'])
            weight_values = layer['weights']
            weight_names = []
            for i in range(len(weight_values)):
                idx = '0' + str(i) if i < 10 else str(i)
                name = 'param_' + idx
                weight_names.append(name.encode('utf8'))
            g.attrs['weight_names'] = weight_names
            for name, val in zip(weight_names, weight_values):
                param_dset = g.create_dataset(name, val.shape, dtype=val.dtype)
                param_dset[:] = val
        f.flush()
        f.close()

    def build(self):
        self.compiled_snn.build()

    def run(self, X_test, Y_test):
        return self.compiled_snn.run(self, X_test, Y_test)

    def export_to_sim(self, path, filename):
        self.compiled_snn.save(path, filename + '_' +
                               self.target_sim.__short_name__)

    def end_sim(self):
        self.compiled_snn.end_sim()
