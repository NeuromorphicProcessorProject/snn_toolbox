"""Testing pooling.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import numpy as np
import theano
import theano.tensor as T

from snntoolbox.simulation.backends.inisim_backend import pool_same_size

x = T.tensor4("x")
y = pool_same_size(x, (2, 2), True, (1, 1))

f = theano.function([x], [y])

A = np.array([[[[1.,  1.,  1.,  1.,  5.,  6.],
                [1.,  1.,  1.,  1.,  11.,  12.],
                [1.,  1.,  1.,  1.,  17.,  18.],
                [1.,  20.,  21.,  22.,  23.,  24.],
                [25.,  26.,  27.,  28.,  29.,  30.],
                [1.,  1.,  1.,  1.,  1.,  1.]]]], dtype=np.float32)

# A = np.arange(36).reshape(1, 1, 6, 6)
# A = np.array(A, dtype=np.float32)

print(f(A))
