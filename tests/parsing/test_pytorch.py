from importlib import import_module

import inspect
import numpy as np
import os
import shutil
from tensorflow.keras import backend

from snntoolbox.bin.utils import initialize_simulator, run_pipeline
from snntoolbox.datasets.utils import get_dataset
from tests.conftest import pytorch_skip_if_dependency_missing
from tests.core.test_models import get_correlations


@pytorch_skip_if_dependency_missing
class TestPytorchParser:
    """Test parsing pytorch models."""

    @staticmethod
    def prepare_model(model, config):
        import torch

        path_wd = config.get('paths', 'path_wd')
        model_name = config.get('paths', 'filename_ann')
        torch.save(model.state_dict(), os.path.join(path_wd,
                                                    model_name + '.pkl'))
        # Need to copy model definition over to temp dir because only weights
        # were saved.
        from tests.parsing.models import pytorch
        source_path = inspect.getfile(pytorch)
        shutil.copyfile(source_path, os.path.join(path_wd, model_name + '.py'))

    def test_loading(self, _model_4, _config):

        assert backend.image_data_format() == 'channels_first', \
            "Pytorch to Keras parser needs image_data_format == channel_first."

        self.prepare_model(_model_4, _config)

        updates = {
            'tools': {'evaluate_ann': True,
                      'parse': False,
                      'normalize': False,
                      'convert': False,
                      'simulate': False},
            'input': {'model_lib': 'pytorch'},
            'simulation': {'num_to_test': 100,
                           'batch_size': 50}}

        _config.read_dict(updates)

        initialize_simulator(_config)

        normset, testset = get_dataset(_config)

        model_lib = import_module('snntoolbox.parsing.model_libs.' +
                                  _config.get('input', 'model_lib') +
                                  '_input_lib')
        input_model = model_lib.load(_config.get('paths', 'path_wd'),
                                     _config.get('paths', 'filename_ann'))

        # Evaluate input model.
        acc = model_lib.evaluate(input_model['val_fn'],
                                 _config.getint('simulation', 'batch_size'),
                                 _config.getint('simulation', 'num_to_test'),
                                 **testset)

        assert acc >= 0.8

    def test_parsing(self, _model_4, _config):

        self.prepare_model(_model_4, _config)

        updates = {
            'tools': {'evaluate_ann': True,
                      'parse': True,
                      'normalize': False,
                      'convert': False,
                      'simulate': False},
            'input': {'model_lib': 'pytorch'},
            'simulation': {'num_to_test': 100,
                           'batch_size': 50}
        }

        _config.read_dict(updates)

        initialize_simulator(_config)

        acc = run_pipeline(_config)

        assert acc[0] >= 0.8

    def test_normalizing(self, _model_4, _config):

        self.prepare_model(_model_4, _config)

        updates = {
            'tools': {'evaluate_ann': True,
                      'parse': True,
                      'normalize': True,
                      'convert': False,
                      'simulate': False},
            'input': {'model_lib': 'pytorch'},
            'simulation': {'num_to_test': 100,
                           'batch_size': 50}
        }

        _config.read_dict(updates)

        initialize_simulator(_config)

        acc = run_pipeline(_config)

        assert acc[0] >= 0.8

    def test_pipeline(self, _model_4, _config):

        self.prepare_model(_model_4, _config)

        updates = {
            'tools': {'evaluate_ann': False},
            'input': {'model_lib': 'pytorch'},
            'simulation': {
                'duration': 100,
                'num_to_test': 100,
                'batch_size': 50},
            'output': {
                'log_vars': {'activations_n_b_l', 'spiketrains_n_b_l_t'}}}

        _config.read_dict(updates)

        initialize_simulator(_config)

        acc = run_pipeline(_config)

        assert acc[0] >= 0.8

        corr = get_correlations(_config)
        assert np.all(corr[:-1] > 0.97)
        assert corr[-1] > 0.5
