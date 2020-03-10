# coding=utf-8
import os
import subprocess
import sys
from importlib import import_module

import keras
import numpy as np

from snntoolbox.bin.utils import initialize_simulator
from snntoolbox.bin.utils import run_pipeline
from snntoolbox.conversion.utils import normalize_parameters
from snntoolbox.datasets.utils import get_dataset
from snntoolbox.simulation.utils import spiketrains_to_rates
from snntoolbox.simulation.plotting import get_pearson_coefficients
from snntoolbox.utils.utils import import_configparser
from tests.conftest import spinnaker_skip_if_dependency_missing
from tests.conftest import nest_skip_if_dependency_missing
from tests.conftest import brian2_skip_if_dependency_missing

configparser = import_configparser()


def get_correlations(config):
    logdir = os.path.join(config.get('paths', 'log_dir_of_current_run'),
                          'log_vars', '0.npz')
    logvars = np.load(logdir, allow_pickle=True)
    spiketrains = logvars['spiketrains_n_b_l_t']
    activations = logvars['activations_n_b_l']

    spikerates = spiketrains_to_rates(
        spiketrains, config.getint('simulation', 'duration'),
        config.get('conversion', 'spike_code'))

    max_rate = 1. / config.getfloat('simulation', 'dt')
    co = get_pearson_coefficients(spikerates, activations, max_rate)
    return np.mean(co, axis=1)


class TestInputModel:
    """Test loading, parsing and evaluating an input ANN model."""

    def test_parsing(self, _model_2, _config):

        # Parsing removes BatchNorm layers, so we make a copy of the model.
        input_model = keras.models.clone_model(_model_2)
        input_model.set_weights(_model_2.get_weights())
        input_model.compile(_model_2.optimizer.__class__.__name__,
                            _model_2.loss, _model_2.metrics)

        num_to_test = 10000
        batch_size = 100
        _config.set('simulation', 'batch_size', str(batch_size))
        _config.set('simulation', 'num_to_test', str(num_to_test))

        _, testset = get_dataset(_config)
        x_test = testset['x_test']
        y_test = testset['y_test']

        model_lib = import_module('snntoolbox.parsing.model_libs.' +
                                  _config.get('input', 'model_lib') +
                                  '_input_lib')
        model_parser = model_lib.ModelParser(input_model, _config)
        model_parser.parse()
        model_parser.build_parsed_model()
        _, acc, _ = model_parser.evaluate(batch_size, num_to_test,
                                          x_test, y_test)
        _, target_acc = _model_2.evaluate(x_test, y_test, batch_size)
        assert acc == target_acc

    def test_normalizing(self, _model_2, _config):

        # Parsing removes BatchNorm layers, so we make a copy of the model.
        input_model = keras.models.clone_model(_model_2)
        input_model.set_weights(_model_2.get_weights())
        input_model.compile(_model_2.optimizer.__class__.__name__,
                            _model_2.loss, _model_2.metrics)

        num_to_test = 10000
        batch_size = 100
        _config.set('simulation', 'batch_size', str(batch_size))
        _config.set('simulation', 'num_to_test', str(num_to_test))

        normset, testset = get_dataset(_config)
        x_test = testset['x_test']
        y_test = testset['y_test']
        x_norm = normset['x_norm']

        model_lib = import_module('snntoolbox.parsing.model_libs.' +
                                  _config.get('input', 'model_lib') +
                                  '_input_lib')
        model_parser = model_lib.ModelParser(input_model, _config)
        model_parser.parse()
        parsed_model = model_parser.build_parsed_model()

        normalize_parameters(parsed_model, _config, x_norm=x_norm)

        _, acc, _ = model_parser.evaluate(batch_size, num_to_test,
                                          x_test, y_test)
        _, target_acc = _model_2.evaluate(x_test, y_test, batch_size)
        assert acc == target_acc


class TestOutputModel:
    """Test building, saving and running the converted SNN model."""

    def test_inisim(self, _model_2, _config):

        path_wd = _config.get('paths', 'path_wd')
        model_name = _config.get('paths', 'filename_ann')
        keras.models.save_model(_model_2,
                                os.path.join(path_wd, model_name + '.h5'))

        updates = {
            'tools': {'evaluate_ann': False},
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

    @brian2_skip_if_dependency_missing
    def test_brian2(self, _model_1, _config):

        path_wd = _config.get('paths', 'path_wd')
        model_name = _config.get('paths', 'filename_ann')
        keras.models.save_model(_model_1,
                                os.path.join(path_wd, model_name + '.h5'))

        updates = {
            'tools': {'evaluate_ann': False},
            'input': {'poisson_input': True},
            'simulation': {
                'simulator': 'brian2',
                'duration': 200,
                'num_to_test': 100,
                'batch_size': 1,
                'dt': 0.1},
            'output': {
                'log_vars': {'activations_n_b_l', 'spiketrains_n_b_l_t'}}}

        _config.read_dict(updates)

        initialize_simulator(_config)

        acc = run_pipeline(_config)

        assert acc[0] >= 0.95

        corr = get_correlations(_config)
        assert np.all(corr[:-1] > 0.97)
        assert corr[-1] > 0.5

    @nest_skip_if_dependency_missing
    def test_nest(self, _model_1, _config):

        path_wd = _config.get('paths', 'path_wd')
        model_name = _config.get('paths', 'filename_ann')
        keras.models.save_model(_model_1,
                                os.path.join(path_wd, model_name + '.h5'))

        updates = {
            'tools': {'evaluate_ann': False},
            'simulation': {
                'simulator': 'nest',
                'duration': 50,
                'num_to_test': 10,
                'batch_size': 1,
                'dt': 0.1},
            'cell': {
                'tau_refrac': 0.1,
                'delay': 0.1,
                'v_thresh': 0.01},
            'output': {
                'log_vars': {'activations_n_b_l', 'spiketrains_n_b_l_t'}}}

        _config.read_dict(updates)

        initialize_simulator(_config)

        acc = run_pipeline(_config)

        assert acc[0] >= 0.95

        corr = get_correlations(_config)
        assert np.all(corr[:-1] > 0.97)
        assert corr[-1] > 0.5

    @spinnaker_skip_if_dependency_missing
    def test_spinnaker(self, _model_1, _config):

        path_wd = _config.get('paths', 'path_wd')
        model_name = _config.get('paths', 'filename_ann')
        keras.models.save_model(_model_1,
                                os.path.join(path_wd, model_name + '.h5'))

        updates = {
            'tools': {'evaluate_ann': False},
            'input': {'poisson_input': True},
            'simulation': {
                'simulator': 'spiNNaker',
                'duration': 100,
                'num_to_test': 1,  # smaller to make more feasible
                'batch_size': 1},
            'output': {
                'log_vars': {'activations_n_b_l', 'spiketrains_n_b_l_t'}}}

        _config.read_dict(updates)

        initialize_simulator(_config)

        acc = run_pipeline(_config)

        assert acc[0] >= 0.95

        corr = get_correlations(_config)
        assert np.all(corr[:-1] > 0.97)
        assert corr[-1] > 0.5

    @spinnaker_skip_if_dependency_missing
    def test_spinnaker_sparse(self, _model_3, _config):

        path_wd = _config.get('paths', 'path_wd')
        model_name = _config.get('paths', 'filename_ann')
        keras.models.save_model(_model_3,
                                os.path.join(path_wd, model_name + '.h5'))

        updates = {
            'tools': {'evaluate_ann': False},
            'input': {'poisson_input': True},
            'simulation': {
                'simulator': 'spiNNaker',
                'duration': 100,
                'num_to_test': 1,  # smaller to make more feasible
                'batch_size': 1},
            'output': {
                'log_vars': {'activations_n_b_l', 'spiketrains_n_b_l_t'}}}

        _config.read_dict(updates)

        initialize_simulator(_config)

        acc = run_pipeline(_config)

        assert acc[0] >= 0.95

        corr = get_correlations(_config)
        assert np.all(corr[:-1] > 0.97)
        assert corr[-1] > 0.5


class TestPipeline:
    """Test complete pipeline for a number of examples."""

    def test_examples(self, _example_filepath):

        returncode = subprocess.call([sys.executable, _example_filepath],
                                     shell=True)
        assert returncode == 0
