# -*- coding: utf-8 -*-

"""
Script to be able to debug the toolbox from within IDE.

Created on Mon Mar  7 15:30:28 2016
@author: rbodo
"""

if __name__ == '__main__':

    from snntoolbox.config import update_setup
    from snntoolbox.core.pipeline import test_full

    settings = {
        'path_wd': '/home/rbodo/.snntoolbox/data/mnist/cnn/lenet5',
        'dataset_path': '/home/rbodo/.snntoolbox/Datasets/mnist/cnn',
        'filename_ann': '98.96',
        'simulator': 'brian',
        'evaluateANN': False,
        'convert': True,
        'duration': 30,
        'batch_size': 2,
        'num_to_test': 10,
        'runlabel': '02',
        'percentile': 99.999,
        'softmax_to_relu': False,
        'log_vars': {'operations_b_t'},
        'plot_vars': {'activations', 'spikerates', 'input_image',
                      'confusion_matrix', 'correlation', 'operations'}
        }

    update_setup(settings)

    test_full()
