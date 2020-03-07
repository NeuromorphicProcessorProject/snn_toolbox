Spiking neural network conversion toolbox
=========================================

The SNN conversion toolbox (SNN-TB) is a framework to transform rate-based
artificial neural networks into spiking neural networks, and to run them using
various spike encodings. A unique feature about SNN-TB is that it accepts input
models from many different deep-learning libraries (Keras / TF, pytorch, ...)
and provides an interface to several backends for simulation (pyNN, brian2,
...) or deployment (SpiNNaker, Loihi).

Please
refer to the `Documentation <http://snntoolbox.readthedocs.io>`_ for a complete
user guide and API reference. See also the accompanying articles
`[Rueckauer et al., 2017] <https://www.frontiersin.org/articles/10.3389/fnins.2017.00682/abstract>`_
and `[Rueckauer and Liu, 2018] <https://ieeexplore.ieee.org/abstract/document/8351295/>`_.
