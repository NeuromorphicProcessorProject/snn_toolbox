# -*- coding: utf-8 -*-
"""
Functions to save various properties of interest in analog and spiking neural
networks to disk.

Created on Wed Nov 18 11:57:29 2015

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import os
from six.moves import cPickle
from snntoolbox import echo
from snntoolbox.config import globalparams

standard_library.install_aliases()


def confirm_overwrite(filepath):
    """
    If globalparams['overwrite']==False and the file exists, ask user if it
    should be overwritten.
    """

    if not globalparams['overwrite'] and os.path.isfile(filepath):
        overwrite = input("[WARNING] {} already exists - ".format(filepath) +
                          "overwrite? [y/n]")
        while overwrite not in ['y', 'n']:
            overwrite = input("Enter 'y' (overwrite) or 'n' (cancel).")
        return overwrite == 'y'
    return True


def save_model(model, path=globalparams['path'],
               filename=globalparams['filename'], spiking=None):
    """
    Write model architecture and weights to disk.

    Parameters
    ----------

    model : network object
        The network model object in the ``model_lib`` language, e.g. keras.
    """

    import snntoolbox

    spiking = False if spiking is None else True

    # Create directory if not existent yet.
    if not os.path.exists(path):
        os.makedirs(path)

    echo("Saving model to {}\n".format(path))

    if globalparams['model_lib'] == 'keras' or \
            snntoolbox._SIMULATOR == 'INI' and spiking:
        filepath = os.path.join(path, filename + '.json')
        if confirm_overwrite(filepath):
            open(filepath, 'w').write(model.to_json())
            model.save_weights(os.path.join(path, filename + '.h5'),
                               overwrite=globalparams['overwrite'])
    elif globalparams['model_lib'] == 'lasagne':
        import lasagne
        params = lasagne.layers.get_all_param_values(model)
        filepath = os.path.join(path, filename + '.h5')
        if confirm_overwrite(filepath):
            save_weights(params, filepath)
    echo("Done.\n")


def save_weights(params, filepath):
    """
    Dump all layer weights to a HDF5 file.
    """

    import h5py

    f = h5py.File(filepath, 'w')

    for i, p in enumerate(params):
        idx = '0' + str(i) if i < 10 else str(i)
        f.create_dataset('param_' + idx, data=p, dtype=p.dtype)
    f.flush()
    f.close()


def save_assembly(assembly):
    """
    Write layers of neural network to disk.

    The size, structure, labels of all the population of an assembly are stored
    in a dictionary such that one can load them again using the
    ``snntoolbox.io.load.load_assembly`` function.

    The term "assembly" refers
    to pyNN internal nomenclature, where ``Assembly`` is a collection of
    layers (``Populations``), which in turn consist of a number of neurons
    (``cells``).

    The database files containing the model architecture will be named
    ``snn_<filename>``.

    Parameters
    ----------

    assembly : list
        List of pyNN ``population`` objects
    """

    filepath = os.path.join(globalparams['path'],
                            'snn_' + globalparams['filename'])

    if not confirm_overwrite(filepath):
        return

    echo("Saving assembly to {}\n".format(filepath))

    s = {}
    labels = []
    variables = ['size', 'structure', 'label']
    for population in assembly:
        labels.append(population.label)
        data = {}
        for variable in variables:
            data[variable] = getattr(population, variable)
        data['celltype'] = population.celltype.describe()
        s[population.label] = data
    s['labels'] = labels  # List of population labels describing the network.
    s['variables'] = variables  # List of variable names.
    s['size'] = len(assembly)  # Number of populations in assembly.
    cPickle.dump(s, open(filepath, 'wb'))
    echo("Done.\n")


def save_connections(projections):
    """
    Write weights of a neural network to disk.

    The weights between two layers are saved in a text file.
    They can then be used to connect pyNN populations e.g. with
    ``sim.Projection(layer1, layer2, sim.FromListConnector(filename))``,
    where ``sim`` is a simulator supported by pyNN, e.g. Brian, NEURON, or
    NEST.

    Parameters
    ----------

    projections : list
        pyNN ``Projection`` objects representing the connections between
        individual layers.

    Return
    ------
    Text files containing the layer connections. Each file is named after the
    layers it connects, e.g. ``layer1_layer2.txt``
    """

    echo("Saving connections to {}\n".format(globalparams['path']))

    # Iterate over layers to save each projection in a separate txt file.
    for projection in projections:
        filepath = os.path.join(globalparams['path'],
                                projection.label.replace('→', '_'))
        if confirm_overwrite(filepath):
            projection.save('connections', filepath)
    echo("Done.\n")


def save_activations(activations):
    """
    Write activations of a neural network to file.
    """

    filepath = os.path.join(globalparams['path'], 'activations')

    if not confirm_overwrite(filepath):
        return

    echo("Saving activations to {}\n".format(filepath))

    cPickle.dump(activations, open(filepath, 'wb'))

    echo("Done.\n")
