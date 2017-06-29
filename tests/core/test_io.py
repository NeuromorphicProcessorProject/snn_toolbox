# coding=utf-8


import numpy as np
from snntoolbox.datasets.utils import get_dataset


class TestGetDataset:
    """Test obtaining the dataset from disk in correct format."""

    def test_get_dataset_from_npz(self, _datapath, _config):
        data = np.random.random_sample((1, 1, 1, 1))
        np.savez_compressed(str(_datapath.join('x_norm')), data)
        np.savez_compressed(str(_datapath.join('x_test')), data)
        np.savez_compressed(str(_datapath.join('y_test')), data)
        _config.set('paths', 'dataset_path', str(_datapath))
        normset, testset = get_dataset(_config)
        assert all([normset, testset])

    def test_get_dataset_from_jpg(self, _datapath, _config):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
            return plt
        classpath = _datapath.mkdir('class_0')
        data = np.random.random_sample((10, 10, 3))
        plt.imsave(str(classpath.join('image_0.jpg')), data)
        _config.read_dict(
            {'paths': {'dataset_path': str(_datapath)},
             'input': {'dataset_format': 'jpg',
                       'dataflow_kwargs': "{'target_size': (11, 12)}",
                       'datagen_kwargs': "{'rescale': 0.003922,"
                                         " 'featurewise_center': True,"
                                         " 'featurewise_std_normalization':"
                                         " True}"}})
        normset, testset = get_dataset(_config)
        assert all([normset, testset])
