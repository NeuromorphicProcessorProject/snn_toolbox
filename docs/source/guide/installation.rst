.. # coding=utf-8

.. _installation:

Installation
============

Release version
---------------

Run ``pip install snntoolbox``. This will install the minimum dependencies
needed to get started. Optional dependencies like matplotlib and scipy enable
generating output plots.

The toolbox relies on Keras internally, and by default installs
`Tensorflow <https://www.tensorflow.org/>`_ as Keras backend. You may want to
update this backend to optimally fit your system.

.. note:: The SNN toolbox provides a built-in simulator to run the converted
   network. This simulator is Keras-based and will use either Tensorflow or
   `Theano <http://www.deeplearning.net/software/theano/install_ubuntu.html>`_
   as backend. Depending on the backend you choose, different features are
   available in the toolbox simulator. You can install both backends and switch
   between them simply by setting the corresponding parameter in the
   :ref:`config file <configuration>`::

      [simulation]
      keras_backend = tensorflow

   As the Theano is not actively developed any more, the tensorflow backend is
   better maintained. The only reason to choose Theano at this point is that we
   provide an implementation of MaxPooling in INIsim, which is not available
   with the tensorflow backend.

.. note:: (March 2020) The toolbox has been tested to work with keras==2.2.5
   and tensorflow==2.0.1.

Development version (recommended)
---------------------------------

To get the latest updates, checkout the `repository <https://github.com/NeuromorphicProcessorProject/snn_toolbox>`_.
In the toolbox root directory ``snn_toolbox/``, run ``pip install .``.

.. note:: Using easy_install via ``python setup.py install`` has been reported
   to fail on some platforms due to dependency issues.

Additional tools
----------------

For testing a converted network, the toolbox includes a ready-to-use spiking
simulator. In addition, you may install and use one of the simulators described
:ref:`here <simulating>`.

.. note:: Depending on the simulator you use, we recommend installing the
   toolbox in a virtual environment, because simulators supported by pyNN may
   require different versions of their dependencies (Brian for instance only
   works with python2).
