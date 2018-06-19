.. # coding=utf-8

.. _installation:

Installation
============

Requirements
------------

First, install `Theano <http://www.deeplearning.net/software/theano/install_ubuntu.html>`_
or `Tensorflow <https://www.tensorflow.org/>`_.

.. note:: The SNN toolbox provides a built-in simulator to run the converted
   network. This simulator is Keras-based and will use either Theano or
   Tensorflow as backend. Depending on the backend you choose, different
   features are available in the toolbox simulator. You can install both
   backends and switch between them simply by setting the corresponding
   parameter in the :ref:`config file <configuration>`::
   
      [simulation]
      keras_backend = tensorflow
  
Release version
---------------

Run ``pip install snntoolbox``. This will install the other minimum dependencies
(Keras, h5py) on the fly.

Development version (recommended)
---------------------------------

To get the latest version, checkout the `repository <https://github.com/NeuromorphicProcessorProject/snn_toolbox>`_.
In the toolbox root directory ``snn_toolbox/``, run ``pip install .``.

.. note:: Do not use ``python setup.py install`` because easy_install caused the
   installation to fail on some platforms due to dependency issues.

Additional tools
----------------

For testing a converted network, the toolbox includes a ready-to-use spiking
simulator. In addition, you may install and use one of the simulators described
:ref:`here <simulating>`.

.. note:: Depending on the simulator you use, we recommend installing the
   toolbox in a virtual environment, because simulators supported by pyNN may
   require different versions of their dependencies (Brian for instance only
   works with python2).
