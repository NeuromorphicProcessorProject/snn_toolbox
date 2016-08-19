# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:17:41 2016

@author: rbodo
"""

import os
import numpy as np


dataset_path = '/home/rbodo/.snntoolbox/datasets/mnist/'
X_train = np.load(os.path.join(dataset_path, 'X_norm.npz'))['arr_0']
idx = np.linspace(0, len(X_train), num=int(len(X_train)/2), dtype=int,
                  endpoint=False)
np.savez_compressed(os.path.join(dataset_path, 'X_norm'), X_train[idx])
