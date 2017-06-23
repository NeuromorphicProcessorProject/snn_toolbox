# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:36:27 2016

@author: rbodo
"""

import h5py
import numpy as np
from importlib import import_module
from snntoolbox.datasets.utils import load_npz
from snntoolbox.conversion.utils import parse, evaluate_keras

# Load Cifar10 dataset
path_to_dataset = '/home/rbodo/.snntoolbox/Datasets/cifar10/binarynet'
X_test = load_npz(path_to_dataset, 'X_test.npz')
Y_test = load_npz(path_to_dataset, 'Y_test.npz')

# Load model
path_to_model = '/home/rbodo/.snntoolbox/data/cifar10/88.22'
filename_model = '88.22'
model_lib = import_module('snntoolbox.model_libs.lasagne_input_lib')
input_model = model_lib.load(path_to_model, filename_model)

# Test accuracy with full precision parameters
model_lib.evaluate(input_model['val_fn'], X_test, Y_test)

# Converted model to simplifed Keras with BatchNormalization layers integrated.
model_parsed = parse(input_model['model'])

# Test parsed model
evaluate_keras(model_parsed, X_test, Y_test)

parameters = model_parsed.get_weights()

parameters_binarized = []
for p in parameters:
    if p.ndim > 1:
        parameters_binarized.append(model_lib.binarize(p))
    else:
        parameters_binarized.append(p)

model_parsed.set_weights(parameters_binarized)

# Load full precision parameters of converted network
path_to_weights = '/home/rbodo/.snntoolbox/data/cifar10/88.22/88.22_parsed.h5'
params_full_prec = []
f = h5py.File(path_to_weights, mode='r')
g = f.get('model_weights')
for k in g.keys():
    h = g.get(k)
    for j in h.keys():
        params_full_prec.append(np.array(h.get(j)))
f.close()
