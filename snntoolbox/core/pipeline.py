# -*- coding: utf-8 -*-
"""
Wrapper script that combines all tools of SNN Toolbox.



@author: rbodo
"""

# For compatibility with python2
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
from importlib import import_module
from future import standard_library
from snntoolbox.core.util import normalize_parameters
from snntoolbox.io_utils.common import get_dataset

standard_library.install_aliases()


def test_full(config, queue=None):
    """Convert an analog network to a spiking network and simulate it.

    Complete pipeline of
        1. loading and testing a pretrained ANN,
        2. normalizing parameters
        3. converting it to SNN,
        4. running it on a simulator,
        5. given a specified hyperparameter range ``params``,
           repeat simulations with modified parameters.

    Parameters
    ----------

    config: configparser.ConfigParser
        ConfigParser containing the user settings.

    queue: Optional[Queue.Queue]
        Results are added to the queue to be displayed in the GUI.

    Returns
    -------

    results: list
        List of the accuracies obtained after simulating with each parameter
        value in config['parameter_sweep']['param_values'].
    """

    batch_size = config.getint('simulation', 'batch_size')
    num_to_test = config.getint('simulation', 'num_to_test')

    # Instantiate an empty spiking network
    target_sim = import_module('snntoolbox.target_simulators.' +
                               config['simulation']['simulator'] +
                               '_target_sim')
    spiking_model = target_sim.SNN(config, queue)

    # ____________________________ LOAD DATASET ______________________________ #

    normset, testset = get_dataset(config)

    if config.getboolean('tools', 'convert') and not is_stop(queue):

        # ___________________________ LOAD MODEL _____________________________ #

        model_lib = import_module('snntoolbox.model_libs.' +
                                  config['input']['model_lib'] + '_input_lib')
        input_model = model_lib.load(config['paths']['path_wd'],
                                     config['paths']['filename_ann'])

        # Evaluate input model.
        if config.getboolean('tools', 'evaluate_ann') and not is_stop(queue):
            print("Evaluating input model on {} samples...".format(num_to_test))
            model_lib.evaluate(input_model['val_fn'], batch_size, num_to_test,
                               **testset)

        # _____________________________ PARSE ________________________________ #

        print("Parsing input model...")
        model_parser = model_lib.ModelParser(input_model['model'], config)
        model_parser.parse()
        parsed_model = model_parser.build_parsed_model()

        # ____________________________ NORMALIZE _____________________________ #

        if config.getboolean('tools', 'normalize') and not is_stop(queue):
            normalize_parameters(parsed_model, config, **normset)

        # Evaluate parsed model.
        if config.getboolean('tools', 'evaluate_ann') and not is_stop(queue):
            print("Evaluating parsed and normalized model on {} samples..."
                  "".format(num_to_test))
            model_parser.evaluate_parsed(batch_size, num_to_test, **testset)

        # Write parsed model to disk
        parsed_model.save(
            os.path.join(config['paths']['path_wd'],
                         config['paths']['filename_parsed_model'] + '.h5'))

        # ____________________________ CONVERT _______________________________ #

        spiking_model.build(parsed_model)

        # Export network in a format specific to the simulator with which it
        # will be tested later.
        spiking_model.save(config['paths']['path_wd'],
                           config['paths']['filename_snn'])

    # _______________________________ SIMULATE _______________________________ #

    if config.getboolean('tools', 'simulate') and not is_stop(queue):

        # Decorate the 'run' function of the spiking model with a parameter
        # sweep function.
        @run_parameter_sweep(config, queue)
        def run(snn, **test_set):
            return snn.run(**test_set)

        # Simulate network
        results = run(spiking_model, **testset)

        # Clean up
        spiking_model.end_sim()

        # Add results to queue to be displayed in GUI.
        if queue:
            queue.put(results)

        return results


def is_stop(queue):
    """Determine if the user pressed 'stop' in the GUI.

    Parameters
    ----------

    queue: Queue.Queue
        Event queue.

    Returns
    -------

    : bool
        ``True`` if user pressed 'stop' in GUI, ``False`` otherwise.
    """

    if not queue:
        return False
    if queue.empty():
        return False
    elif queue.get_nowait() == 'stop':
        print("Skipped step after user interrupt")
        queue.put('stop')
        return True


def run_parameter_sweep(config, queue):
    """
    Decorator to perform a parameter sweep using the ``run_single`` function.
    Need an aditional wrapping layer to be able to pass decorator arguments.
    """

    def decorator(run_single):

        from functools import wraps

        @wraps(run_single)
        def wrapper(snn, **testset):

            from snntoolbox.io_utils.plotting import plot_param_sweep

            results = []
            param_values = eval(config['parameter_sweep']['param_values'])
            param_name = config['parameter_sweep']['param_name']
            param_logscale = config.getboolean('parameter_sweep',
                                               'param_logscale')
            if len(param_values) > 1:
                print("Testing SNN for parameter values {} = ".format(
                    param_name))
                print(['{:.2f}'.format(i) for i in param_values])
                print('\n')

            # Loop over parameter to sweep
            for p in param_values:
                if is_stop(queue):
                    break

                # Display current parameter value
                config.set('cell', param_name, str(p))
                if len(param_values) > 1:
                    print("\nCurrent value of parameter to sweep: " +
                          "{} = {:.2f}\n".format(param_name, p))

                results.append(run_single(snn, **testset))

            # Plot and return results of parameter sweep.
            plot_param_sweep(
                results, config.getint('simulation', 'num_to_test'),
                param_values, param_name, param_logscale)
            return results
        return wrapper
    return decorator
