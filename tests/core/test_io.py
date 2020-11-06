# coding=utf-8
import os

import numpy as np
from tensorflow import keras

from snntoolbox.bin.utils import run_pipeline
from snntoolbox.datasets.utils import get_dataset


class TestGetDataset:
    """Test obtaining the dataset from disk in correct format."""

    def test_get_dataset_from_npz(self, _config):
        normset, testset = get_dataset(_config)
        assert all([normset, testset])

    def test_get_dataset_from_png(self, _config):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        datapath = _config.get('paths', 'dataset_path')
        classpath = os.path.join(datapath, 'class_0')
        os.mkdir(classpath)
        data = np.random.random_sample((10, 10, 3))
        plt.imsave(os.path.join(classpath, 'image_0.png'), data)

        _config.read_dict({
             'input': {'dataset_format': 'png',
                       'dataflow_kwargs': "{'target_size': (11, 12)}",
                       'datagen_kwargs': "{'rescale': 0.003922,"
                                         " 'featurewise_center': True,"
                                         " 'featurewise_std_normalization':"
                                         " True}"}})

        normset, testset = get_dataset(_config)
        assert all([normset, testset])


class TestModelIO:
    """Test saving and loading various model instances along the pipeline."""

    def test_loading_parsed_model(self, _model_1, _config):

        model = _model_1
        path_wd = _config.get('paths', 'path_wd')
        model_name = _config.get('paths', 'filename_ann')
        keras.models.save_model(model, os.path.join(path_wd,
                                                    model_name + '.h5'))

        # Only perform parsing step.
        updates = {'tools': {'evaluate_ann': False, 'normalize': False,
                             'convert': False, 'simulate': False}}

        _config.read_dict(updates)

        run_pipeline(_config)

        path_parsed = os.path.join(path_wd, _config.get(
            'paths', 'filename_parsed_model') + '.h5')
        model_parsed = keras.models.load_model(path_parsed)

        x = np.random.random_sample((5,) + model.input_shape[1:])
        y = model.predict(x)
        y_parsed = model_parsed.predict(x)

        assert np.allclose(y, y_parsed)
