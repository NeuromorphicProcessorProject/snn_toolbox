|b1| |b2| |b3| |b4|

.. |b1| image:: https://travis-ci.org/NeuromorphicProcessorProject/snn_toolbox.svg?branch=master
    :target: https://travis-ci.org/NeuromorphicProcessorProject/snn_toolbox

.. |b2| image:: https://readthedocs.org/projects/snntoolbox/badge/?version=latest
    :target: https://snntoolbox.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |b3| image:: https://badge.fury.io/py/snntoolbox.svg
    :target: https://badge.fury.io/py/snntoolbox

.. |b4| image:: https://pepy.tech/badge/snntoolbox
    :target: https://pepy.tech/project/snntoolbox
    

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
`[Rueckauer et al., 2017] <https://www.frontiersin.org/articles/10.3389/fnins.2017.00682/abstract>`_, `[Rueckauer and Liu, 2018] <https://ieeexplore.ieee.org/abstract/document/8351295/>`_, and `[Rueckauer and Liu, 2021] <https://ieeexplore.ieee.org/abstract/document/9533837>`_.
