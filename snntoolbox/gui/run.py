# -*- coding: utf-8 -*-

"""
Script to use the toolbox from console instead of GUI.

Created on Mon Mar  7 15:30:28 2016
@author: rbodo
"""

import json
from snntoolbox.config import update_setup
from snntoolbox.core.pipeline import test_full
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

class_idx_path = '/home/rbodo/.snntoolbox/Datasets/imagenet/imagenet_class_index_1008.json'
class_idx = json.load(open(class_idx_path, 'r'))
classes = [class_idx[str(idx)][0] for idx in range(len(class_idx))]

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


settings = {'path_wd': '/home/rbodo/.snntoolbox/data/imagenet/inceptionV3/lasagne/averagepool',
            'dataset_path': '/home/rbodo/.snntoolbox/Datasets/imagenet/validation',
            'dataset_format': 'jpg',
            'datagen_kwargs': {'preprocessing_function': preprocess_input},
            'dataflow_kwargs': str({'target_size': (299, 299), 'classes': classes,
                                    'shuffle': False}),
            'filename_ann': '69.70_89.38',
            'model_lib': 'lasagne',
            'evaluateANN': True,
            'duration': 200,
            'batch_size': 1,
            'num_to_test': 1,
            'runlabel': 'base',
            'percentile': 99.999,
            'log_vars': {'operations_b_t'},
            'plot_vars': {'activations', 'spikerates', 'input_image',
                          'confusion_matrix', 'correlation', 'operations'}
            }

update_setup(settings)

test_full()
