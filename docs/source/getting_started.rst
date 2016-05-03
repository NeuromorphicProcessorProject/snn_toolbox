Getting Started
===============

Installation
------------

We recommend using virtual environments for different simulators, since the simulators 
supported by pyNN may require different versions of python, numpy, ...
(Brian for instance only works with python-2).

Requirements
............

* Install `Keras <keras.io>`_. (In a later version, this toolbox will support alternative
  neural network libraries like Torch, Cafe or Lasagne.)
* ``pip install freetype`` (including ``libpng``)
* If you plan to get the development version, install the packages listed in ``requirements.txt``.

Prebuild version
................

* Download the archive ``dist/snntoolbox-<version>-py2.py3-none-any.whl``
* Run ``pip install snntoolbox-<version>-py2.py3-none-any.whl``,
  which will get all other dependencies for you and install Brian as default simulator.
  Add other simulators as needed.

Development version
...................

* To get the latest version, checkout the `repository <https://code.ini.uzh.ch/NPP_theory/SNN_toolbox.git>`_
* The toolbox includes a ready-to-use simulator developed at INI.
  In addition, you may install a simulator supported by `pyNN <http://neuralensemble.org/docs/PyNN/>`_,
  or bring your own custom simulator that accepts a pyNN model as input.
* Run some :doc:`tests </tests>` to try the complete pipeline of loading, converting
  and simulating a network.


Examples - Fully Connected Network on MNIST
-------------------------------------------

Normally, we would run the toolbox simply by specifying a set of parameters and
then calling :py:func:`tests.util.test_full`, like this:

.. code-block:: python

	import snntoolbox

    # Define parameters
    globalparams = {'dataset': 'mnist',  # Dataset
                    'architecture': 'mlp',  # Model type
                    'filename': '98.05',  # Name of file containing the model
                    'evaluateANN': True,  # Test accuracy of input model
                    'normalize': True,  # Normalize weights before conversion
                    'sim_only': False,  # Skip conversion and simulate SNN directly
                    'verbose': 2}  # Show plots and temporary results
    cellparams = {'v_thresh': 1.0,  # Threshold potential
                  'v_reset': 0.0}  # Reset potential
    simparams = {'duration': 100.0,  # Simulation time
                 'num_to_test': 2}  # Samples to evaluate
    
    # Run network (including loading the model, weight normalization, conversion
    # and simulation)
    snntoolbox.update_setup(globalparams, cellparams, simparams)
    snntoolbox.test_full()


However, here are three usecases that allow some more insight into the application of this toolbox:

	A. `Conversion only`_
	B. `Simulation only`_
	C. `Parameter sweep`_

For a description of ``global_params``, ``cell_params``, and ``sim_params``,
see :doc:`configure_toolbox`.

.. _Conversion only:
.. _spiking network:

Usecase A - Conversion only
...........................

Pipeline:
	1. Load and test a pretrained ANN
	2. Normalize weights
	3. Convert to SNN
	4. Save SNN to disk

.. code-block:: python

    # For compatibility with python2
    from __future__ import print_function, unicode_literals
    from __future__ import division, absolute_import
    from future import standard_library

    from SNN_toolbox import sim
    from SNN_toolbox.config import update_setup, globalparams
    from SNN_toolbox.core.conversion import convert_to_SNN
    from SNN_toolbox.core.normalization import normalize_weights
    from SNN_toolbox.io.load import load_model, get_reshaped_dataset, ANN
    from SNN_toolbox.tests.util import evaluate

    standard_library.install_aliases()

    # Parameters
    global_params = {'dataset': 'mnist',
                     'architecture': 'cnn',
                     'path': '../data/',
                     'filename': '99.06'}

    # Check that parameter choices are valid. Parameters that were not
    # specified above are filled in from the default parameters.
    update_setup(global_params=global_params)

    sim.setup()

    # Load dataset, reshaped according to network architecture
    (X_train, Y_train, X_test, Y_test) = get_reshaped_dataset()

    # Load model structure and weights
    model = load_model()

    # Evaluate ANN before normalization to ensure it doesn't affect accuracy
    score = evaluate(model, X_test, Y_test, **{'show_accuracy': True})
    print('\n Before weight normalization:')
    print('Test score: {:.2f}'.format(score[0]))
    print('Test accuracy: {:.2%} \n'.format(score[1]))

    # Normalize ANN
    model = normalize_weights(model,
                              X_train[:int(len(X_train) *
                                      globalparams['fracNorm']), :],
                              globalparams['path'])

    # Re-evaluate ANN
    score = evaluate(model, X_test, Y_test, **{'show_accuracy': True})
    print('Test score: {:.2f}'.format(score[0]))
    print('Test accuracy: {:.2%} \n'.format(score[1]))

    # Extract architecture and weights from model.
    ann = ANN(model)

    # Compile spiking network from ANN. SNN is written to
	# <path>/<dataset>/<architecture>/<filename>/<simulator>.
    convert_to_SNN(ann)


.. _Simulation only:
.. _evaluated:

Usecase B - Simulation only
...........................

Pipeline:
	1. Specify parameters
	2. Load dataset
	3. Call ``run_SNN``. This will

		- load your already converted SNN
		- run it on a spiking simulator
		- Plot spikerates, spiketrains and membrane voltage.

It is assumed that a network has been converted using for instance the script
``convert_only.py``. (There should be a folder in
``<repo_root>/<path>/<dataset>/<architecture>/`` containing the converted
network.)

.. code-block:: python

    # For compatibility with python2
    from __future__ import print_function, unicode_literals
    from __future__ import division, absolute_import
    from future import standard_library

    from SNN_toolbox.config import update_setup
    from SNN_toolbox.io.load import get_reshaped_dataset
    from SNN_toolbox.core.simulation import run_SNN

    standard_library.install_aliases()

    # Parameters
    global_params = {'dataset': 'mnist',
                     'architecture': 'cnn',
                     'path': '../data/',
                     'filename': '99.06'}
    cell_params = {'v_thresh': 1.0,
                   'v_reset': 0.0}
    sim_params = {'duration': 1000.0,
                  'dt': 10,
                  'num_to_test': 2}

    # Check that parameter choices are valid. Parameters that were not
    # specified above are filled in from the default parameters.
    update_setup(global_params, cell_params, sim_params)

    # Load dataset, reshaped according to network architecture
    (X_train, Y_train, X_test, Y_test) = get_reshaped_dataset()

    # Simulate spiking network
    run_SNN(X_test, Y_test)


.. _Parameter sweep:

Usecase C - Parameter sweep
...........................

Pipeline:
	1. Specify parameters
	2. Define a parameter range to sweep, e.g. for `v_thresh`
	3. Call ``test_full``. This will

		- load an already converted SNN
		- run it repeatedly on a spiking simulator while varying the hyperparameter
		- plot accuracy vs. hyperparameter

.. code-block:: python

    # For compatibility with python2
    from __future__ import print_function, unicode_literals
    from __future__ import division, absolute_import
    from future import standard_library

    from SNN_toolbox.tests.util import get_range, test_full
    from SNN_toolbox.config import update_setup

    standard_library.install_aliases()

    # Parameters
    global_params = {'dataset': 'mnist',
                     'architecture': 'cnn',
                     'path': '../data/',
                     'filename': '99.06',
                     'sim_only': True}  # This skips loading, normalizing and converting the ann
    cell_params = {'v_reset': 0.0}
    sim_params = {'duration': 100.0,
                  'dt': 5.0,
                  'num_to_test': 2}

    update_setup(global_params=global_params,
                 cell_params=cell_params,
                 sim_params=sim_params)

    # Define parameter values to sweep
    thresholds = get_range(0.4, 1.5, 2, method='linear')

    # Run simulation for each value in the specified parameter range.
    # The method `test_full` combines and generalizes loading, normalization,
    # evaluation, conversion and simulation steps. It also plots accuracy vs
    # hyperparameter.
    (results, spiketrains, vmem) = test_full(thresholds, 'v_thresh')



Contact
-------

* Bodo Rueckauer


