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
import theano.tensor as t
import numpy as np

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

standard_library.install_aliases()


# Our own rounding function, that does not set the gradient to 0 like Theano's
class Round3(UnaryScalarOp):
    """Rounding function.

    """

    def R_op(self, inputs, eval_points):
        """

        Parameters
        ----------

        inputs :
        eval_points :
        """

        pass

    def c_code(self, node, name, xx, zz, sub):
        """

        Parameters
        ----------

        node :
        name :
        xx :
        zz :
        sub :

        Returns
        -------


        """

        (x,) = xx
        (z,) = zz
        return "%(z)s = round(%(x)s);" % locals()

    def grad(self, inputs, gout):
        """

        Parameters
        ----------

        inputs :
        gout :

        Returns
        -------


        """

        (gz,) = gout
        return gz,

round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)


def hard_sigmoid(x):
    """

    Parameters
    ----------

    x :

    Returns
    -------

    """

    return t.clip((x + 1.) / 2., 0, 1)


# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    """

    Parameters
    ----------

    x :

    Returns
    -------


    """

    return 2.*round3(hard_sigmoid(x))-1.


def binary_sigmoid_unit(x):
    """

    Parameters
    ----------

    x :

    Returns
    -------


    """

    return round3(hard_sigmoid(x))


# The weights' binarization function,
# taken directly from the BinaryConnect github repository
# (which was made available by his authors)

def binarization(w, h, binary=True, deterministic=False, stochastic=False):
    """

    Parameters
    ----------

    w :
    h :
    binary :
    deterministic :
    stochastic :

    Returns
    -------


    """

    if not binary or (deterministic and stochastic):
        print("not binary")
        wb = w

    else:

        # [-1,1] -> [0,1]
        wb = hard_sigmoid(w/h)
        # wb = t.clip(w/h,-1,1)

        # Stochastic BinaryConnect
        if stochastic:

            print("stoch")
            wb = t.cast(np.random.binomial(1, wb, t.shape(wb)),
                        theano.config.floatX)

        # Deterministic BinaryConnect (round to nearest)
        else:
            print("det")
            wb = t.round(wb)

        # 0 or 1 -> -1 or 1
        wb = t.cast(t.switch(wb, h, -h), theano.config.floatX)

    return wb
