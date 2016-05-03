# -*- coding: utf-8 -*-
"""
Usecase:
1. load and test a pretrained ANN
2. normalize weights
3. converting to SNN
4. save SNN to disk

This example uses MNIST dataset and a fully-connected network.

Created on Wed Feb 17 09:45:22 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

from snntoolbox import sim
from snntoolbox.config import update_setup, globalparams
from snntoolbox.core.conversion import convert_to_SNN
from snntoolbox.core.normalization import normalize_weights
from snntoolbox.io.load import load_model, get_reshaped_dataset, ANN
from snntoolbox.core.util import evaluate

standard_library.install_aliases()


if __name__ == '__main__':

    # Parameters
    global_params = {'dataset': 'mnist',
                     'architecture': 'cnn',
                     'path': '../data/',
                     'filename': '99.06'}

    # Check that parameter choices are valid. Parameters that were not
    # specified above are filled in from the default parameters.
    update_setup(global_params=global_params)

    sim.setup()

    # Load dataset, reshaped according to network architecture
    (X_train, Y_train, X_test, Y_test) = get_reshaped_dataset()

    # Load model structure and weights
    model = load_model('ann_' + globalparams['filename'])

    # Evaluate ANN before normalization to ensure it doesn't affect accuracy
    score = evaluate(model, X_test, Y_test, **{'show_accuracy': True})
    print('\n Before weight normalization:')
    print('Test score: {:.2f}'.format(score[0]))
    print('Test accuracy: {:.2%} \n'.format(score[1]))

    # Normalize ANN
    model = normalize_weights(model,
                              X_train[:int(len(X_train) *
                                      globalparams['fracNorm']), :],
                              globalparams['path'])

    # Re-evaluate ANN
    score = evaluate(model, X_test, Y_test, **{'show_accuracy': True})
    print('Test score: {:.2f}'.format(score[0]))
    print('Test accuracy: {:.2%} \n'.format(score[1]))

    # Extract architecture and weights from model.
    ann = ANN(model)

    # Compile spiking network from ANN. SNN is written to
    # <path>/<dataset>/<architecture>/<filename>/.
    convert_to_SNN(ann)
