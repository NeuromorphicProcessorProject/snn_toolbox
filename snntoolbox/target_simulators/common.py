# coding=utf-8

"""Common functions for spiking simulators."""

import numpy as np


def get_samples_from_list(x_test, y_test, dataflow, settings):
    """
    If user specified a list of samples to test with
    ``settings['sample_indices_to_test']``, this function extract them from the
    test set.
    """

    si = settings['sample_indices_to_test'].copy() \
        if 'sample_indices_to_test' in settings else []
    if not si == []:
        if dataflow is not None:
            batch_idx = 0
            x_test = []
            y_test = []
            target_idx = si.pop(0)
            while len(x_test) < settings['num_to_test']:
                x_b_l, y_b = dataflow.next()
                for i in range(settings['batch_size']):
                    if batch_idx * settings['batch_size'] + i == target_idx:
                        x_test.append(x_b_l[i])
                        y_test.append(y_b[i])
                        if len(si) > 0:
                            target_idx = si.pop(0)
                batch_idx += 1
            x_test = np.array(x_test)
            y_test = np.array(y_test)
        else:
            x_test = np.array([x_test[i] for i in si])
            y_test = np.array([y_test[i] for i in si])

    return x_test, y_test
