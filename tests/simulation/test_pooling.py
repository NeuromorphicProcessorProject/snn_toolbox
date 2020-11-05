import os
import numpy as np
import pytest
from tensorflow.keras import models
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense

from snntoolbox.bin.utils import run_pipeline
from tests.core.test_models import get_correlations


@pytest.fixture(scope='session')
def _model_maxpool2D_1(_dataset):
    """Patch size 2, stride 2."""

    x_train, y_train, x_test, y_test = _dataset

    input_shape = x_train.shape[1:]
    input_layer = Input(input_shape)

    layer = MaxPooling2D()(input_layer)
    layer = Flatten()(layer)
    layer = Dense(units=10,
                  activation='softmax',
                  use_bias=False)(layer)

    model = Model(input_layer, layer)

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

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
            'simulation': {
                'duration': 100,
                'num_to_test': 100,
                'batch_size': 50},
            'output': {
                'log_vars': {'activations_n_b_l', 'spiketrains_n_b_l_t'}}}

        _config.read_dict(updates)

        acc = run_pipeline(_config)

        assert acc[0] >= 0.95

        corr = get_correlations(_config)
        assert np.all(corr[:-1] > 0.99)
        assert corr[-1] > 0.90
