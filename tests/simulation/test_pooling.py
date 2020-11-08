import os
import numpy as np
import pytest
from tensorflow.keras import Input, Model, models, backend
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense

from snntoolbox.bin.utils import run_pipeline
from tests.core.test_models import get_correlations, get_ann_acc

DATA_FORMAT = backend.image_data_format()


@pytest.fixture(scope='session')
def _model_maxpool2D_1(_dataset):
    return get_model_maxpool2D_1(_dataset)


@pytest.fixture(scope='session')
def _model_maxpool2D_1_first(_dataset_first):
    backend.set_image_data_format('channels_first')
    model = get_model_maxpool2D_1(_dataset_first)
    backend.set_image_data_format(DATA_FORMAT)
    return model


@pytest.fixture(scope='session')
def _model_maxpool2D_1_last(_dataset_last):
    backend.set_image_data_format('channels_last')
    model = get_model_maxpool2D_1(_dataset_last)
    backend.set_image_data_format(DATA_FORMAT)
    return model


def get_model_maxpool2D_1(dataset):
    """Patch size 2, stride 2."""

    x_train, y_train, x_test, y_test = dataset

    input_shape = x_train.shape[1:]
    input_layer = Input(input_shape)

    layer = MaxPooling2D()(input_layer)
    layer = Flatten()(layer)
    layer = Dense(10, activation='softmax', use_bias=False)(layer)

    model = Model(input_layer, layer)

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    history = model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2,
                        validation_data=(x_test, y_test))

    assert history.history['val_accuracy'][-1] > 0.8

    return model


class TestMaxPoolingINI:
    """Test spiking max-pooling layers in INIsim, using tensorflow backend."""

    def test_maxpool_fallback(self, _model_maxpool2D_1, _config):
        """Test that maxpooling falls back on average pooling."""
        path_wd = _config.get('paths', 'path_wd')
        model_name = _config.get('paths', 'filename_ann')
        models.save_model(_model_maxpool2D_1,
                          os.path.join(path_wd, model_name + '.h5'))

        updates = {
            'tools': {'evaluate_ann': False, 'normalize': False},
            'conversion': {'max2avg_pool': True},
            'simulation': {
                'duration': 100,
                'num_to_test': 100,
                'batch_size': 50},
            'output': {
                'log_vars': {'activations_n_b_l', 'spiketrains_n_b_l_t'}}}

        _config.read_dict(updates)

        acc = run_pipeline(_config)

        assert acc[0] >= 0.8

        corr = get_correlations(_config)
        assert np.all(corr[:-1] > 0.99)
        assert corr[-1] > 0.90

    reason = "tf.ops.max_pool only works with NHWC data format"

    @pytest.mark.skip(reason)
    @pytest.mark.xfail(raises=ValueError, reason=reason)
    def test_maxpool_first(self, _model_maxpool2D_1_first, _config_first):
        """Test that maxpooling fails with data format channels_first."""
        config = _config_first
        path_wd = config.get('paths', 'path_wd')
        model_name = config.get('paths', 'filename_ann')
        models.save_model(_model_maxpool2D_1_first,
                          os.path.join(path_wd, model_name + '.h5'))

        updates = {
            'tools': {'evaluate_ann': False, 'normalize': False},
            'simulation': {
                'duration': 100,
                'num_to_test': 100,
                'batch_size': 50},
            'output': {
                'log_vars': {'activations_n_b_l', 'spiketrains_n_b_l_t'}}}

        config.read_dict(updates)

        run_pipeline(config)

    def test_maxpool(self, _model_maxpool2D_1_last, _config_last):
        """Test maxpooling."""
        config = _config_last
        path_wd = config.get('paths', 'path_wd')
        model_name = config.get('paths', 'filename_ann')
        models.save_model(_model_maxpool2D_1_last,
                          os.path.join(path_wd, model_name + '.h5'))

        updates = {
            'tools': {'evaluate_ann': False, 'normalize': False},
            'simulation': {
                'duration': 100,
                'num_to_test': 100,
                'batch_size': 50},
            'output': {
                'log_vars': {'activations_n_b_l', 'spiketrains_n_b_l_t'}}}

        config.read_dict(updates)

        acc = run_pipeline(config)

        acc_ann = get_ann_acc(config)
        assert acc[0] >= 0.9 * acc_ann

        corr = get_correlations(config)
        assert np.all(corr[:-1] > 0.99)
        assert corr[-1] > 0.90
