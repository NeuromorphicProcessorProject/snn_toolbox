.. -*- coding=utf-8 -*-

Spiking neural network conversion toolbox
=========================================

The SNN conversion toolbox (SNN-TB) is a framework to transform rate-based
artificial neural networks into spiking neural networks, and to run them using
various spike encodings. A unique feature about SNN-TB is that it accepts input
models from many different deep-learning libraries (Keras / TF, pytorch, ...)
and provides an interface to several backends for simulation (pyNN, brian2,
...) or deployment (SpiNNaker, Loihi). The source code can be found on
`GitHub`_. See also the accompanying articles
`[Rueckauer et al., 2017] <https://www.frontiersin.org/articles/10.3389/fnins.2017.00682/abstract>`_,
`[Rueckauer and Liu, 2018] <https://ieeexplore.ieee.org/abstract/document/8351295/>`_, and `[Rueckauer and Liu, 2021] <https://ieeexplore.ieee.org/abstract/document/9533837>`_.

User Guide
----------

These sections guide you through the installation, configuration and running of
the toolbox. Examples are included.

.. toctree::
   :maxdepth: 1

   guide/intro
   guide/installation
   guide/running
   guide/configuration
   guide/extending
   guide/examples
   guide/citation
   guide/support

API Reference
-------------

Here you find detailed descriptions of specific functions and classes.

.. toctree::
   :maxdepth: 2

   api/bin
   api/parsing
   api/conversion
   api/simulation
   api/datasets
   api/utils

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/NeuromorphicProcessorProject/snn_toolbox
