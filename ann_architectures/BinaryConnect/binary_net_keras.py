# coding=utf-8

"""BinaryNet implemented in Keras."""

import keras
from ann_architectures.BinaryConnect.common import binarization


# This class extends the Lasagne DenseLayer to support BinaryConnect
class Dense(keras.layers.Dense):
    """Binary Dense layer."""

    def __init__(self, num_units, binary=True, stochastic=True, h=1., **kwargs):

        self.binary = binary
        self.stochastic = stochastic
        self.H = h
        self.W = None
        self.Wb = None

        if self.binary:
            super(Dense, self).__init__(num_units, init='uniform', **kwargs)
        else:
            super(Dense, self).__init__(num_units, **kwargs)

    def get_output_for(self, input_node, deterministic=False, **kwargs):
        """

        Parameters
        ----------
        input_node :
        deterministic :
        kwargs :

        Returns
        -------

        """

        self.Wb = binarization(self.W, self.H, self.binary, deterministic,
                               self.stochastic)
        wr = self.W
        self.W = self.Wb

        rvalue = super(Dense, self).get_output_for(input_node, **kwargs)

        self.W = wr

        return rvalue


# This class extends the Lasagne Conv2DLayer to support BinaryConnect
class Convolution2D(keras.layers.Convolution2D):
    """Binary convolution layer."""

    def __init__(self, nb_filter, nb_row, nb_col, binary=True, stochastic=True,
                 h=1., **kwargs):

        self.binary = binary
        self.stochastic = stochastic
        self.H = h
        self.W = None
        self.Wb = None

        if self.binary:
            super(Convolution2D, self).__init__(nb_filter, nb_row, nb_col,
                                                init='uniform', **kwargs)
        else:
            super(Convolution2D, self).__init__(nb_filter, nb_row, nb_col,
                                                **kwargs)

    def convolve(self, input_node, deterministic=False, **kwargs):
        """Convolution operation.

        Parameters
        ----------
        input_node :
        deterministic :
        kwargs :

        Returns
        -------

        """

        self.Wb = binarization(self.W, self.H, self.binary, deterministic,
                               self.stochastic)
        wr = self.W
        self.W = self.Wb

        rvalue = super(Convolution2D, self).convolve(input_node, **kwargs)

        self.W = wr

        return rvalue
