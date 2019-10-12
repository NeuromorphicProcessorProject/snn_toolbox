# coding=utf-8
import os

import numpy as np

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
