# -*- coding: utf-8 -*-
"""Class for neural networks that have been converted from analog to spiking.

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

    Instances of this class contains all essential information about a network,
    independently of the model library in which the original network was built
    (e.g. Keras). This makes the SNN toolbox stable against changes in input
    formats. Another advantage is extensibility: In order to add a new input
    language to the toolbox (e.g. Caffe), a developer only needs to add a
    single module to ``model_libs`` package, implementing a number of methods
    (see the respective functions in 'keras_input_lib.py' for more details.)

    Parameters
    ----------

    path: string, optional
        Path to directory where to load model from. Defaults to
        ``settings['path']``.

    filename: string, optional
        Name of file to load model from. Defaults to ``settings['filename']``.

    Attributes
    ----------

    model: Model
        A model instance of the network in the respective ``model_lib``.

    val_fn: Theano function
        A Theano function that allows evaluating the original model.

    input_shape: list
        The dimensions of the input sample:
        [batch_size, n_chnls, n_rows, n_cols]. For instance, mnist would have
        input shape [Null, 1, 28, 28].

    layers: list
        List of all the layers of the network, where each layer contains a
        dictionary with keys

        - layer_num (int): Index of layer.
        - layer_type (string): Describing the type, e.g. `Dense`,
          `Convolution`, `Pool`.
        - output_shape (list): The output dimensions of the layer.

        In addition, `Dense` and `Convolution` layer types contain

        - parameters (array): The weights and biases connecting this layer with
          the previous.

        `Convolution` layers contain further

        - nb_col (int): The x-dimension of filters.
        - nb_row (int): The y-dimension of filters.
        - border_mode (string): How to handle borders during convolution, e.g.
          `full`, `valid`, `same`.

        `Pooling` layers contain

        - pool_size (list): Specifies the subsampling factor in each dimension.
        - strides (list): The stepsize in each dimension during pooling.

        `Activation` layers (including Pooling) contain

        - get_activ: A Theano function computing the activations of a layer.

    labels: list
        The layer labels.

    layer_idx_map: list
        A list mapping the layer indices of the original network to the parsed
        network. (Not all layers of the original model are needed in the parsed
        model.) For instance: To get the layer index i of the original input
        ``model`` that corresponds to layer j of the parsed network ``layers``,
        one would use ``i = layer_idx_map[j]``.

    compiled_snn: SNN_compiled
        Object containing the compiled spiking network (ready for simulation).

    """

    def __init__(self, path=None, filename=None):
        """Init a SNN model."""
        if path is None:
            path = settings['path']
        if filename is None:
            filename = settings['filename']

        # Import utility functions of input model library ('model_lib') and
        # of the simulator to use ('target_sim')
        self.import_modules()

        # Load input model structure and parameters.
        model = self.model_lib.load_ann(path, filename)
        self.model = model['model']
        self.val_fn = model['val_fn']

        # Parse input model to our common format, extracting all necessary
        # information about layers.
        ann = self.model_lib.extract(model)
        self.input_shape = ann['input_shape']
        self.layers = ann['layers']
        self.labels = ann['labels']
        self.layer_idx_map = ann['layer_idx_map']

        # Allocate an object which will contain the compiled spiking network
        # (ready for simulation)
        self.compiled_snn = self.target_sim.SNN_compiled(ann)

    def import_modules(self):
        """Import utility functions of input model library.

        Import utility functions of input model library ('model_lib') and of
        the simulator to use ('target_sim')

        """
        from importlib import import_module
        self.model_lib = import_module('snntoolbox.model_libs.' +
                                       settings['model_lib'] + '_input_lib')
        self.target_sim = import_module('snntoolbox.target_simulators.' +
                                        settings['simulator'] + '_target_sim')

    def evaluate_ann(self, X_test, Y_test, **kwargs):
        """Evaluate the performance of a network.

        Wrapper for the evaluation functions of specific input neural network
        libraries ``settings['model_lib']`` like keras, caffe, torch, etc.

        Parameters
        ----------

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
        score of a Keras model.

        """
        score = self.model_lib.evaluate(self.val_fn, X_test, Y_test)

        print('\n')
        print("Test score: {:.2f}".format(score[0]))
        print("Test accuracy: {:.2%}\n".format(score[1]))

        return score

    def normalize_parameters(self):
        """Normalize the parameters of a network.

        The parameters of each layer are normalized with respect to the maximum
        activation or parameter value.

        """
        from snntoolbox.io_utils.plotting import plot_hist
        from snntoolbox.core.util import get_scale_fac, get_activations_layer
        from snntoolbox.io_utils.load import load_dataset

        print("Loading normalization data set.\n")
        X_norm = load_dataset(settings['dataset_path'], 'X_norm.npz')  # t=0.2%
        print("Using {} samples for normalization.".format(len(X_norm)))

#        import numpy as np
#        sizes = [len(X_norm) * np.array(layer['output_shape'][1:]).prod() *
#                 32 / (8 * 1e9) for idx, layer in enumerate(self.layers)
#                 if idx != 0 and 'parameters' in self.layers[idx-1]]
#        size_str = ['{:.2f}'.format(s) for s in sizes]
#        print("INFO: Need {} GB for layer activations.\n".format(size_str) +
#              "May have to reduce size of data set used for normalization.\n")

        print("Normalizing parameters:\n")
        newpath = os.path.join(settings['log_dir_of_current_run'],
                               'normalization')
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        # Loop through all layers, looking for layers with parameters
        scale_fac_prev_layer = 1
        for idx, layer in enumerate(self.layers):
            # Skip layer if not preceeded by a layer with parameters
            if idx == 0 or 'parameters' not in self.layers[idx-1].keys():
                continue
            print("Calculating output of activation layer {}".format(idx) +
                  " following layer {} with shape {}...".format(
                  self.labels[idx-1], layer['output_shape']))
            parameters = self.layers[idx-1]['parameters']
            # Undo previous scaling before calculating activations:
            self.set_layer_params([parameters[0] * scale_fac_prev_layer,
                                   parameters[1]], idx-1)
            # t=4.9%
            activations = get_activations_layer(layer['get_activ'], X_norm)
            if settings['normalization_schedule']:
                scale_fac = get_scale_fac(activations, idx)
            else:
                scale_fac = get_scale_fac(activations)  # t=3.7%
            parameters_norm = [
                parameters[0] * scale_fac_prev_layer / scale_fac,
                parameters[1] / scale_fac]
            scale_fac_prev_layer = scale_fac
            # Update model with modified parameters
            self.set_layer_params(parameters_norm, idx-1)
            if settings['verbose'] < 3:
                continue
            weight_dict = {
                'weights': parameters[0].flatten(),
                'weights_norm': parameters_norm[0].flatten()}
            # t=2.8%
            plot_hist(weight_dict, 'Weight', self.labels[idx-1], newpath)

            if True:  # Too costly
                continue
            # Compute activations with modified parameters
            activations_norm = get_activations_layer(layer['get_activ'],
                                                     X_norm)  # t=4.8%
            activation_dict = {'Activations': activations.flatten(),
                               'Activations_norm': activations_norm.flatten()}
            plot_hist(activation_dict, 'Activation', self.labels[idx-1],
                      newpath, scale_fac)  # t=83.1%

    def model_based_normalize(self):
        """Model based normalization."""
        scale_fac_prev_layer = 1
        for idx, layer in enumerate(self.layers):
            if idx == 0 or 'parameters' not in self.layers[idx-1].keys():
                continue
            print("Calculating output of activation layer {}".format(idx) +
                  " following layer {} with shape {}...".format(
                  self.labels[idx-1], layer['output_shape']))
            parameters = self.layers[idx-1]['parameters']
            # Undo previous scaling before calculating activations:
            self.set_layer_params([parameters[0] * scale_fac_prev_layer,
                                   parameters[1]], idx-1)

            # TODO: normalize based on maximum weights in the layer.

    def set_layer_params(self, parameters, i):
        """Set ``parameters`` of layer ``i``."""
        self.layers[i]['parameters'] = parameters
        self.model_lib.set_layer_params(self.model, parameters,
                                        self.layer_idx_map[i])

    def get_params(self):
        """Return list where each entry contains the parameters of a layer."""
        return [l['parameters'] for l in self.layers
                if 'parameters' in l.keys()]

    def save(self, path=None, filename=None):
        """Write model architecture and parameters to disk.

        Parameters
        ----------

        path: string, optional
            Path to directory where to save model. Defaults to
            ``settings['path']``.

        filename: string, optional
            Name of file to write model to. Defaults to
            ``settings['filename_snn']``.

        """
        from snntoolbox.io_utils.save import confirm_overwrite

        if path is None:
            path = settings['path']
        if filename is None:
            filename = settings['filename_snn']

        print("Saving parsed model to {}...".format(path))
        # Create directory if not existent yet.
        if not os.path.exists(path):
            os.makedirs(path)

        filepath = os.path.join(path, filename)
        if confirm_overwrite(filepath + '.h5'):
            self.save_parameters(filepath + '.h5')

        if confirm_overwrite(filepath + '.json'):
            self.save_config(filepath + '.json')

        print("Done.\n")

    def get_config(self):
        """Return a dictionary describing the model."""
        # When adding a parser for a new input model format, you may need to
        # supplement the following list by non-JSON-serializable objects.
        skip_keys = ['parameters', 'get_activ', 'model', 'model_protobuf']
        layer_config = []
        for layer in self.layers:
            layer_config.append([{key: layer[key]} for key in layer.keys()
                                 if key not in skip_keys])
        return {'name': self.__class__.__name__,
                'input_shape': self.input_shape,
                'layers': layer_config}

    def save_config(self, path=None):
        """Save model configuration to disk."""
        from snntoolbox.io_utils.save import to_json

        if path is None:
            path = settings['path']

        to_json(self.get_config(), path)

    def save_parameters(self, path=None):
        """Dump all layer parameters to a HDF5 file."""
        import h5py

        if path is None:
            path = settings['path']

        f = h5py.File(path, 'w')

        f.attrs['layer_names'] = [l.encode('utf8') for l in self.labels]

        for layer in self.layers:
            if 'parameters' not in layer.keys():
                continue
            g = f.create_group(layer['label'])
            param_values = layer['parameters']
            param_names = []
            for i in range(len(param_values)):
                idx = '0' + str(i) if i < 10 else str(i)
                name = 'param_' + idx
                param_names.append(name.encode('utf8'))
            g.attrs['param_names'] = param_names
            for name, val in zip(param_names, param_values):
                param_dset = g.create_dataset(name, val.shape, dtype=val.dtype)
                param_dset[:] = val
        f.flush()
        f.close()

    def build(self):
        """Build model."""
        self.compiled_snn.build()

    def run(self, X_test, Y_test):
        """Run model."""
        return self.compiled_snn.run(self, X_test, Y_test)

    def export_to_sim(self, path=None, filename=None):
        """Export converted SNN."""
        if path is None:
            path = settings['path']
        if filename is None:
            filename = settings['filename_snn_exported']

        self.compiled_snn.save(path, filename)

    def end_sim(self):
        """End simulation."""
        self.compiled_snn.end_sim()
