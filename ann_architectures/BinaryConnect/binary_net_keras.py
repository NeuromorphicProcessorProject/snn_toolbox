from builtins import super
import keras
from ann_architectures.BinaryConnect.common import binarization


# This class extends the Lasagne DenseLayer to support BinaryConnect
class Dense(keras.layers.Dense):

    def __init__(self, num_units, binary=True, stochastic=True, H=1.,
                 **kwargs):

        self.binary = binary
        self.stochastic = stochastic

        if self.binary:
            super().__init__(num_units, init='uniform', **kwargs)
        else:
            super().__init__(num_units, **kwargs)

    def get_output_for(self, input, deterministic=False, **kwargs):

        self.Wb = binarization(self.W, self.H, self.binary, deterministic,
                               self.stochastic)
        Wr = self.W
        self.W = self.Wb

        rvalue = super(Dense, self).get_output_for(input, **kwargs)

        self.W = Wr

        return rvalue


# This class extends the Lasagne Conv2DLayer to support BinaryConnect
class Convolution2D(keras.layers.Convolution2D):

    def __init__(self, nb_filter, nb_row, nb_col, binary=True, stochastic=True,
                 H=1., **kwargs):

        self.binary = binary
        self.stochastic = stochastic

        if self.binary:
            super().__init__(nb_filter, nb_row, nb_col, init='uniform',
                             **kwargs)
        else:
            super().__init__(nb_filter, nb_row, nb_col, **kwargs)

    def convolve(self, input, deterministic=False, **kwargs):

        self.Wb = binarization(self.W, self.H, self.binary, deterministic,
                               self.stochastic)
        Wr = self.W
        self.W = self.Wb

        rvalue = super(Convolution2D, self).convolve(input, **kwargs)

        self.W = Wr

        return rvalue
