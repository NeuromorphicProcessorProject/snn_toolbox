.. # coding=utf-8

.. _extending:

Extending the toolbox
=====================

Input side: Adding a new model library
--------------------------------------

The philosophy behind the toolbox architecture is to make all steps in the
conversion/simulation pipeline independent of the original model format.
Therefore, in order to add a new input model library (e.g. Torch) to the
toolbox, put a module named ``torch_input_lib`` into the
:py:mod:`snntoolbox.parsing.model_libs` package. Then create a child class
``torch_input_lib.ModelParser`` inheriting from
`snntoolbox.parsing.utils.AbstractModelParser`, and implement the abstract
methods tailored to the new input model library.

Output side: Adding a custom simulator
--------------------------------------

Similarly, adding another simulator to run converted networks implies adding a
file to the :py:mod:`snntoolbox.simulation.target_simulators` package. Each
file in there allows building a spiking network and exporting it for use in a
specific spiking simulator.

To add a simulator called 'custom', put a file named ``<custom>_target_sim.py``
into :py:mod:`~snntoolbox.simulation.target_simulators`. Then create a child
class ``SNN`` inheriting from `AbstractSNN`, and implement the abstract methods
tailored to the 'custom' simulator.

Adding custom layers to an existing simulator
---------------------------------------------

Say you want to add support for a new layer class to the toolbox e.g. you want
to be able to use locally connected layers in INIsim. You would have to do the
following:

    - Register the new layer class in the ``[restrictions]`` section of
      :py:mod:`snntoolbox.config_defaults`. Otherwise the parser will remove
      the layer.
    - Add a method to the parser so the layer gets ported correctly from your
      input model library to the internal Keras model. See e.g.
      :py:meth:`this example <snntoolbox.parsing.model_libs.keras_input_lib.ModelParser.parse_dense>`
      for the Keras parser.
    - Define a spiking version of your new layer in the simulation / deployment
      backend of your choice, e.g. like :py:class:`here <snntoolbox.simulation.backends.inisim.temporal_mean_rate_tensorflow.SpikeDense>`.
      Also, in case of INIsim, register the new class in the ``custom_layers``
      dict :py:const:`snntoolbox/simulation/backends/inisim/temporal_mean_rate_tensorflow.py:748`.
      If necessary, you can add special behavior when building this new layer:
      :py:meth:`snntoolbox.simulation.target_simulators.INI_temporal_mean_rate_target_sim.SNN.build_dense`.
    - If you are interested in the number of operations your SNN consumes, add
      a corresponding clause in :py:func:`snntoolbox.parsing.utils.get_fanout`
      that defines the synaptic connectivity.
