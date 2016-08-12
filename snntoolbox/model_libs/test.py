# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:22:05 2016

@author: rbodo
"""

import numpy as np
np.random.seed(2)

w = np.ones((2, 4, 3, 3))
#w = np.ones((4, 2))
b = np.ones(2)

beta = np.random.randint(10, size=4)
gamma = np.random.randint(10, size=4)
mean = np.random.randint(10, size=4)
std = np.random.randint(10, size=4)
epsilon = 1e-6

axis_in = 1 if w.ndim > 2 else 0
axis_out = 0 if w.ndim > 2 else 1
reduction_axes = list(range(w.ndim))
del reduction_axes[axis_out]

broadcast_shape = [1] * w.ndim
broadcast_shape[axis_in] = w.shape[axis_in]
mean = np.reshape(mean, broadcast_shape)
std = np.reshape(std, broadcast_shape)
beta = np.reshape(beta, broadcast_shape)
gamma = np.reshape(gamma, broadcast_shape)

b += np.sum(w * (beta - mean * gamma / (std + epsilon)), axis=tuple(reduction_axes))
w *= gamma / (std + epsilon)
