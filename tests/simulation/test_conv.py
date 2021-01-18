import os
import numpy as np
import pytest
from tensorflow.keras import Input, Model, models, backend
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Reshape, \
    BatchNormalization, ReLU

from snntoolbox.bin.utils import run_pipeline
from tests.core.test_models import get_correlations, get_ann_acc

DATA_FORMAT = backend.image_data_format()


@pytest.fixture(scope='session')
def _model_conv1D_1(_dataset):
    return get_model_conv1D_1(_dataset)


@pytest.fixture(scope='session')
def _model_conv1D_1_last(_dataset_last):
    backend.set_image_data_format('channels_last')
    model = get_model_conv1D_1(_dataset_last)
    backend.set_image_data_format(DATA_FORMAT)
    return model


def get_model_conv1D_1(dataset):
    x_train, y_train, x_test, y_test = dataset

    axis = 1 if backend.image_data_format() == 'channels_first' else 2
    input_shape = x_train.shape[1:]
    input_shape_1d = (input_shape[0] * input_shape[1], input_shape[2])
    input_layer = Input(input_shape)
    layer = Reshape(input_shape_1d)(input_layer)
    layer = Conv1D(8, 1)(layer)
    layer = BatchNormalization(axis)(layer)
    layer = ReLU()(layer)
    layer = Conv1D(16, 3, strides=4, padding='causal')(layer)
    layer = BatchNormalization(axis)(layer)
    layer = ReLU()(layer)
    layer = Conv1D(32, 2, padding='causal', dilation_rate=2)(layer)
    layer = BatchNormalization(axis)(layer)
    layer = ReLU()(layer)
    layer = Flatten()(layer)
    layer = Dense(10, activation='softmax', use_bias=False)(layer)

    model = Model(input_layer, layer)

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    history = model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2,
                        validation_data=(x_test, y_test))

    assert history.history['val_accuracy'][-1] > 0.8

    return model


class TestConv1dINI:
    """Test spiking conv1d layers in INIsim, using tensorflow backend."""

    def test_conv1d(self, _model_conv1D_1_last, _config_last):
        """Test conv1d."""
        config = _config_last
        path_wd = config.get('paths', 'path_wd')
        model_name = config.get('paths', 'filename_ann')
        models.save_model(_model_conv1D_1_last,
                          os.path.join(path_wd, model_name + '.h5'))

        updates = {
            'tools': {'evaluate_ann': False, 'normalize': True},
            'simulation': {
                'duration': 200,
                'num_to_test': 100,
                'batch_size': 50},
            'output': {
                'log_vars': {'activations_n_b_l', 'spiketrains_n_b_l_t'},
            }}

        config.read_dict(updates)

        acc = run_pipeline(config)

        acc_ann = get_ann_acc(config)
        assert acc[0] >= 0.98 * acc_ann

        corr = get_correlations(config)
        assert np.all(corr[:-1] > 0.97)
        assert corr[-1] > 0.90
