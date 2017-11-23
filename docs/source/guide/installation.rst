.. # coding=utf-8

.. _installation:

Installation
============

Requirements
------------

First, install `Theano <http://www.deeplearning.net/software/theano/install_ubuntu.html>`_.
Note: As of 22.11.2017, the latest Keras version was not compatible with Theano
1.0, so please install Theano version 0.9.

Stable version
--------------

Run ``pip install snntoolbox``. This will install the other minimum dependencies
(Keras, h5py) on the fly.

Development version
-------------------

To get the latest version, checkout the `repository <https://github.com/NeuromorphicProcessorProject/snn_toolbox>`_.
In the toolbox root directory ``snn_toolbox/``, run ``pip install .``. Do not
use ``python setup.py install`` because easy_install caused the installation to
fail on some platforms due to dependency issues.

Additional tools
----------------

For testing a converted network, the toolbox includes a ready-to-use spiking
simulator. In addition, you may install and use one of the simulators described
:ref:`here <simulating>`.

Depending on the simulator you use, we recommend installing the toolbox in a
virtual environment, because simulators supported by pyNN may require different
versions of their dependencies (Brian for instance only works with python2).
