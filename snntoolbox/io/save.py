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


def save_assembly(assembly, path, filename):
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

    filepath = os.path.join(path, 'snn_' + filename)

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


def save_connections(projections, path):
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

    echo("Saving connections to {}\n".format(path))

    # Iterate over layers to save each projection in a separate txt file.
    for projection in projections:
        filepath = os.path.join(path, projection.label.replace('→', '_'))
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
