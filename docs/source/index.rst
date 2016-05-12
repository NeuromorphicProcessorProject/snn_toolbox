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

.. figure:: workflow.png
   :scale: 50 %
   :alt: Workflow diagram of the SNN toolbox.

   **SNN toolbox workflow.** The input network (e.g. a Keras model) is transformed into an instance of the
   ``ANN`` class. The ``core.conversion`` module turns this into a spiking network. Finally, the 
   network can be evaluated in any spiking simulator that supports pyNN_ as input. At any stage of the 
   pipeline, models and results can be written to disk (see :py:mod:`io.save` in :doc:`modules`).


Features
--------

* Before conversion, the input model is transcribed into a custom class :py:class:`io.load.ANN`
  containing only essential model structure and weights in common python containers.
  The conversion toolbox currently supports input networks generated with Keras or Lasagne.
  See :doc:`getting_started` on how to extend the relevant classes like ``ANN`` to handle models from other 
  common libraries like caffe, torch etc.
* The resulting spiking network is given in `pyNN <http://neuralensemble.org/docs/PyNN/>`_.
  The toolbox allows running the spiking network in any simulator supported by pyNN
  (currently `Brian <http://briansimulator.org/>`_, `Nest <http://www.nest-simulator.org/>`_,
  `Neuron <https://www.neuron.yale.edu/neuron/>`_), or by a custom simulator that allows pyNN models as inputs.
* In addition to supporting the simulators listed above, the toolbox includes a ready-to-use
  simulator developed at INI. This simulator features a very simple integrate-and-fire neuron.
  By dispensing with redundant parameters and implementing a highly parallel simulation, the run time
  is reduced by several orders of magnitude.
* The weights of analog neural networks can be normalized for achieving higher accuracy
  in the converted net.
* Examples for both convolutional networks and fully-connected networks on MNIST and CIFAR10 are provided.
* So far, this toolbox is able to handle classification datasets. For other applications,
  the ``io.load.get_dataset`` module needs to be extended.

.. figure:: snntoolbox_gui.png
   :scale: 50 %
   :alt: Snapshot of the SNN toolbox GUI.

   **SNN toolbox GUI.** On the left hand side, the user can specify which tools to use during the experiment (e.g. whether or not to normalize weights prior to conversion, to evaluate the ANN before converting, to load an already converted net and simulate only, etc.). Also, parameters of the neuron cells used during simulation can be set. On the right hand side, the results of a test run can be plotted and examined for each layer of the network separately. The example above compares ANN activations to SNN spikerates for the first convolutional layer on the MNIST dataset. The GUI saves and reloads last settings automatically, and allows saving and loading preferences manually. Tooltips explain all functionality.

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

