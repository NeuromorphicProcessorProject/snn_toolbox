.. # coding=utf-8

Extending the toolbox
=====================

Input side: Adding a new model library
--------------------------------------

The philosophy behind the toolbox architecture is to make all steps in the
conversion/simulation pipeline independent of the original model format.
Therefore, in order to add a new input model library (e.g. Torch) to the
toolbox, put a module named ``torch_input_lib`` into the `model_libs` package.
Then create a child class `lasagne_input_lib.ModelParser` inheriting from
`AbstractModelParser`, and implement the abstract methods tailored to the new
input model library.

Output side: Adding a custom simulator
--------------------------------------

Similarly, adding another simulator to run converted networks implies adding a
file to the `target_simulators` package. Each file in there allows building a
spiking network and exporting it for use in a specific spiking simulator.

To add a simulator called 'custom', put a file named ``<custom>_target_sim.py``
into `target_simulators`. Then create a child class ``SNN`` inheriting from
`AbstractSNN`, and implement the abstract methods tailored to the 'custom'
simulator.

Requested extensions
--------------------

* Add support for TensorFlow input models.
