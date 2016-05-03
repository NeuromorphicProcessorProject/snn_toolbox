.. SNN toolbox documentation master file, created by
   sphinx-quickstart on Sun Mar 13 07:59:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the SNN toolbox documentation!
=========================================

This is a toolbox for converting analog to spiking neural networks (ANN to SNN),
and running them in a spiking neuron simulator.

Citation
--------

::

	Diehl, P.U. and Neil, D. and Binas, J. and Cook, M. and Liu, S.C. and Pfeiffer, M.
	Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing,
	IEEE International Joint Conference on Neural Networks (IJCNN), 2015


Features
--------

* Before conversion, the input model is transcribed into a custom class :py:class:`io.load.ANN`
  containing only essential model structure and weights in common python containers.
  The conversion toolbox currently supports input networks generated with keras,
  but extending the ``ANN`` class to extract models from other common libraries
  like caffe, torch etc. is straightforward.
* The resulting spiking network is given in `pyNN <http://neuralensemble.org/docs/PyNN/>`_.
  The toolbox allows running the spiking network in any simulator supported by pyNN
  (currently `Brian <http://briansimulator.org/>`_, `Nest <http://www.nest-simulator.org/>`_, `Neuron <https://www.neuron.yale.edu/neuron/>`_), or by a custom simulator that allows pyNN models as inputs.
* In addition to supporting the simulators listed above, the toolbox includes a ready-to-use
  simulator developed at INI. This simulator features a very simple integrate-and-fire neuron.
  By dispensing with redundant parameters and implementing a highly parallel simulation, the run time
  is reduced by several orders of magnitude.
* The weights of analog neural networks can be normalized for achieving higher accuracy
  in the converted net.
* Examples for both convolutional networks and fully-connected networks on MNIST and CIFAR10 are provided.
* So far, this toolbox is able to handle classification datasets. For other applications,
  the ``io.load.get_dataset`` module needs to be extended.

.. figure:: workflow.png
   :scale: 50 %
   :alt: Workflow diagram of the SNN toolbox.

   **SNN toolbox workflow.** The input network (e.g. a Keras model) is transformed into an instance of the
   ``ANN`` class. The ``core.conversion`` module turns this into a spiking network. Finally, the 
   network can be evaluated in any spiking simulator that supports pyNN_ as input. At any stage of the 
   pipeline, models and results can be written to disk (see :py:mod:`io.save` in :doc:`modules`).

.. toctree::
   :maxdepth: 3

   getting_started
   configure_toolbox
   modules
   Tests <tests>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

