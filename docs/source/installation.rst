.. # coding=utf-8

Installation
============

Requirements
------------

First, install `Theano <http://www.deeplearning.net/software/theano/install_ubuntu.html>`_.

Stable version
--------------

Run ``pip install snntoolbox``.

This will install the other minimum dependencies (Keras, matplotlib, h5py).

For testing a converted network, the toolbox includes a ready-to-use spiking
simulator. In addition, you may install and use one of the simulators described
:ref:`here <simulating>`.

We recommend installing the toolbox in a virtual environment, because simulators
supported by pyNN may require different versions of their dependencies; Brian
for instance only works with python2.

Development version
-------------------

* To get the latest version, checkout the `repository <git@github.com:NeuromorphicProcessorProject/snn_toolbox.git>`_.
* In the toolbox root directory ``snn_toolbox/``, run ``python setup.py install``.

