# coding=utf-8

"""Common functions for spiking simulators."""

import numpy as np


def get_samples_from_list(x_test, y_test, dataflow, config):
    """
    If user specified a list of samples to test with
    ``settings['sample_idxs_to_test']``, this function extract them from the
    test set.
    """

    si = eval(config['simulation']['sample_idxs_to_test'])
    if not si == []:
        if dataflow is not None:
            batch_idx = 0
            x_test = []
            y_test = []
            target_idx = si.pop(0)
            while len(x_test) < config.getint('simulation', 'num_to_test'):
                x_b_l, y_b = dataflow.next()
                for i in range(config.getint('simulation', 'batch_size')):
                    if batch_idx * config.getint('simulation', 'batch_size') \
                            + i == target_idx:
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
