# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:50:33 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import theano
import theano.tensor as T
import numpy as np

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

# specifying the gpu to use
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')

standard_library.install_aliases()


# Our own rounding function, that does not set the gradient to 0 like Theano's
class Round3(UnaryScalarOp):

    def c_code(self, node, name, xx, zz, sub):
        (x,) = xx
        (z,) = zz
        return "%(z)s = round(%(x)s);" % locals()

    def grad(self, inputs, gout):
        (gz,) = gout
        return gz,

round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)


def hard_sigmoid(x):
    return T.clip((x+1.)/2., 0, 1)


# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    return 2.*round3(hard_sigmoid(x))-1.


def binary_sigmoid_unit(x):
    return round3(hard_sigmoid(x))


# The weights' binarization function,
# taken directly from the BinaryConnect github repository
# (which was made available by his authors)
def binarization(W, H, binary=True, deterministic=False, stochastic=False):

    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        print("not binary")
        Wb = W

    else:

        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        # Wb = T.clip(W/H,-1,1)

        # Stochastic BinaryConnect
        if stochastic:

            print("stoch")
            Wb = T.cast(np.random.binomial(1, Wb, T.shape(Wb)),
                        theano.config.floatX)

        # Deterministic BinaryConnect (round to nearest)
        else:
            print("det")
            Wb = T.round(Wb)

        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb, H, -H), theano.config.floatX)

    return Wb
