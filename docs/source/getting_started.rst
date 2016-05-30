Getting Started
===============

Installation
------------

We recommend using virtual environments for different simulators, since the simulators 
supported by pyNN may require different versions of python, numpy, ...
(Brian for instance only works with python-2).

Requirements
............

* Install `Theano <http://www.deeplearning.net/software/theano/>`_.
* All other dependencies will be installed automatically.
* For testing a converted network, the toolbox includes a ready-to-use spiking
  simulator developed at INI. In addition, you may install a simulator supported
  by `pyNN <http://neuralensemble.org/docs/PyNN/>`_, or bring your own custom
  simulator that accepts a pyNN model as input.

Prebuild version
................

* Download the archive ``dist/snntoolbox-<version>-py2.py3-none-any.whl``
* Run ``pip install snntoolbox-<version>-py2.py3-none-any.whl``.

Development version
...................

* To get the latest version, checkout the `repository <git@github.com:dannyneil/chimera_sim.git>`_
* In the toolbox root directory ``SNN_toolbox/``, run ``python setup.py develop``.

Running the toolbox
-------------------

In a terminal window, type ``snntoolbox`` to start the main GUI containing all tools.

Alternitively, read and run example.py in snntoolbox/tests/, which contains a number of typical usecases.

Extending the toolbox
---------------------

Have a look at the :doc:`pipeline </core/pipeline>` module to examine the complete pipeline of

    1. loading and testing a pretrained ANN,
    2. normalizing weights
    3. converting it to SNN,
    4. running it on a simulator,
    5. if given a specified hyperparameter range ``params``,
       repeat simulations with modified parameters.

Input side: Adding a new model library
......................................

So far, the toolbox supports input models written in Keras and Lasagne.

The philosophy behind the architecture is to make all steps in the conversion/simulation
pipeline independent of the original model format. Therefore, in order to add a
new input model library (e.g. Caffe) to the toolbox, put a module named ``caffe_input_lib``
into the ``model_libs`` package. Have a look at one of the existing files there to get an idea
what functions have to be implemented. The return requirements are specified in their
respective docstrings. Basically, all it needs is a function to parse the essential
information about layers into a common format used by the toolbox further down the line.

Output side: Adding a custom simulator
......................................

Similarly, adding another simulator to run converted networks implies adding a file to the
``target_simulators`` package. Each file in there allow building a spiking network
and exporting it for use in a specific spiking simulator.

To add a simulator called 'custom', put a file named ``custom_target_sim.py`` into ``target_simulators``. Then implement the class ``SNN_compiled`` with its
methods (``load``, ``save``, ``build``, ``run``) tailored to 'custom' simulator.

Examples - Fully Connected Network on MNIST
-------------------------------------------

Normally, we would run the toolbox simply by typing ``snntoolbox`` in the terminal
and using the GUI.

If working with a python interpreter, one would specify a set of parameters and
then call :py:func:`tests.util.test_full`, like this:

.. code-block:: python

    import snntoolbox

    # Define parameters
    settings = {'dataset': 'mnist',
                'architecture': 'cnn',
                'filename': '99.16',
                'path': 'example/mnist/cnn/99.16/INI/',
                'evaluateANN': True,  # Test accuracy of input model
                'normalize': True,  # Normalize weights to get better perf.
                'convert': True,  # Convert analog net to spiking
                'simulate': True,  # Simulate converted net
                'verbose': 3,  # Show plots and temporary results
                'v_thresh': 1,  # Threshold potential
                'simulator': 'INI',  # Reset potential
                'duration': 100.0}  # Simulation time
    
    # Run network (including loading the model, weight normalization, conversion
    # and simulation)
    snntoolbox.update_setup(settings)
    snntoolbox.test_full()


However, here are three usecases that allow some more insight into the application of this toolbox:

    A. `Conversion only`_
    B. `Simulation only`_
    C. `Parameter sweep`_

For a description of the possible values for the parameters in ``settings``,
see :doc:`configure_toolbox`.

.. _Conversion only:
.. _spiking network:

Usecase A - Conversion only
...........................

Steps:
    1. Set ``convert = True`` and ``simulate = False``
    2. Specify other parameters (working directory, filename, ...)
    3. Update settings: ``update_setup(settings)``
    4. Call ``test_full()``. This will

        - load the dataset,
        - load a pretrained ANN from ``<path>/<filename>``
        - optionally evaluate it (``evaluate = True``),
        - optionally normalize weights (``normalize = True``),
        - convert to spiking,
        - save SNN to disk.

.. _Simulation only:
.. _evaluated:

Usecase B - Simulation only
...........................

Steps:
    1. Set ``convert = False`` and ``simulate = True``
    2. Specify other parameters (working directory, simulator to use, ...)
    3. Update settings: ``update_setup(settings)``
    4. Call ``test_full()``. This will

        - load the dataset,
        - load your already converted SNN,
        - run the net on a spiking simulator,
        - plot spikerates, spiketrains, activations, correlations, etc.

    Note: It is assumed that a network has already been converted (e.g. with
    Usecase A). I.e. there should be a folder in ``<path>`` containing the
    converted network, named ``snn_<filename>_<simulator>``.

.. _Parameter sweep:

Usecase C - Parameter sweep
...........................

Steps:
    1. Specify parameters and update settings with ``update_setup(settings)``
    2. Define a parameter range to sweep, e.g. for `v_thresh`, using for
       instance the helper function ``get_range()``
    3. Call ``test_full``. This will

        - load an already converted SNN or perform a conversion as specified in
          settings.
        - run the SNN repeatedly on a spiking simulator while varying the
          hyperparameter
        - plot accuracy vs. hyperparameter

Usecase C is shown in full in the example below.

.. code-block:: python

    import snntoolbox

    # Parameters
    settings = {'dataset': 'mnist',
                'architecture': 'cnn',
                'filename': '99.16',
                'path': 'example/mnist/cnn/99.16/INI/',
                'evaluateANN': True,  # Test accuracy of input model
                'normalize': True,  # Normalize weights to get better perf.
                'convert': True,  # Convert analog net to spiking
                'simulate': True,  # Simulate converted net
                'verbose': 3,  # Show plots and temporary results
                'v_thresh': 1,  # Threshold potential
                'simulator': 'INI',  # Reset potential
                'duration': 100.0}  # Simulation time
    
    # Update defaults with parameters specified above:
    snntoolbox.update_setup(settings)
    
    # Run network (including loading the model, weight normalization,
    # conversion and simulation).
    
    # If set True, the converted model is simulated for three different values
    # of v_thresh. Otherwise use parameters as specified above,
    # for a single run.
    do_param_sweep = True
    if do_param_sweep:
        param_name = 'v_thresh'
        params = snntoolbox.get_range(0.1, 1.5, 3, method='linear')
        snntoolbox.test_full(params=params,
                             param_name=param_name,
                             param_logscale=False)
    else:
        snntoolbox.test_full()



Contact
-------

* Bodo Rueckauer


