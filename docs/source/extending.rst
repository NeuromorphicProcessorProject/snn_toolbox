.. # coding=utf-8

Extending the toolbox
=====================

Input side: Adding a new model library
--------------------------------------

Currently, the toolbox supports input models written in Keras, Lasagne and Caffe.

The philosophy behind the toolbox architecture is to make all steps in the conversion/simulation
pipeline independent of the original model format. Therefore, in order to add a
new input model library (e.g. Torch) to the toolbox, put a module named ``torch_input_lib``
into the ``model_libs`` package. Have a look at one of the existing files there
to get an idea what functions have to be implemented. The return requirements
are specified in their respective docstrings. Basically, all it needs is a
function to parse the essential information about layers into a common format
used by the toolbox further down the line.

Output side: Adding a custom simulator
--------------------------------------

Similarly, adding another simulator to run converted networks implies adding a file to the
``target_simulators`` package. Each file in there allows building a spiking network
and exporting it for use in a specific spiking simulator.

To add a simulator called 'custom', put a file named ``<custom>_target_sim.py``
into ``target_simulators``. Then implement the class ``SNN`` with its methods
(``load``, ``save``, ``build``, ``run``) tailored to 'custom' simulator.

Requested extensions
--------------------

This is a TODO-list to lift current limitations of the toolbox.

* Implement nonzero biases for pyNN and Brian2 simulators (working for INIsim).
  Currently, biases in our pyNN export are included as offset currents, but
  this does not seem to have any effect...
* In pyNN and Brian2, implement analog input currents instead of Poisson input (works in
  INIsim)

