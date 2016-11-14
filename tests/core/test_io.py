# coding=utf-8


import numpy as np
from snntoolbox.core.util import get_dataset


class TestGetDataset:
    """Test obtaining the dataset from disk in correct format."""

    def test_get_dataset_from_npz(self, datapath, settings):
        data = np.random.random_sample((1, 1, 1, 1))
        np.savez_compressed(str(datapath.join('x_norm')), data)
        np.savez_compressed(str(datapath.join('x_test')), data)
        np.savez_compressed(str(datapath.join('y_test')), data)
        settings['dataset_path'] = str(datapath)
        evalset, normset, testset = get_dataset(settings)
        assert all([evalset, normset, testset])

    def test_get_dataset_from_jpg(self, datapath, settings):
        import matplotlib.pyplot as plt
        classpath = datapath.mkdir('class_0')
        data = np.random.random_sample((10, 10, 3))
        plt.imsave(str(classpath.join('image_0.jpg')), data)
        settings.update(
            {'dataset_format': 'jpg',
             'dataset_path': str(datapath),
             'dataflow_kwargs': "{'target_size': (11, 12)}",
             'datagen_kwargs': "{'rescale': 0.003922,"
                               " 'featurewise_center': True,"
                               " 'featurewise_std_normalization': True}"})
        evalset, normset, testset = get_dataset(settings)
        assert all([evalset, normset, testset])
