# coding=utf-8


class TestInputModel:
    """Test loading, parsing and evaluating an input ANN model."""

    def test_loading(self, _input_model_and_lib, _testset, _config):
        batch_size = _config.getint('simulation', 'batch_size')
        num_to_test = _config.getint('simulation', 'num_to_test')
        model_lib = _input_model_and_lib['model_lib']
        input_model = _input_model_and_lib['input_model']
        score = model_lib.evaluate(input_model['val_fn'], batch_size,
                                   num_to_test, **_testset)
        target_acc = _input_model_and_lib['target_acc']
        assert round(100 * score, 2) >= target_acc

    def test_parsing(self, _input_model_and_lib, _testset, _config):
        batch_size = _config.getint('simulation', 'batch_size')
        num_to_test = _config.getint('simulation', 'num_to_test')
        model_parser = _input_model_and_lib['model_lib'].ModelParser(
            _input_model_and_lib['input_model']['model'], _config)
        model_parser.parse()
        model_parser.build_parsed_model()
        score = model_parser.evaluate(batch_size, num_to_test, **_testset)
        target_acc = _input_model_and_lib['target_acc']
        assert round(100 * score[1], 2) >= target_acc

    def test_normalizing(self, _input_model_and_lib, _normset, _testset,
                         _config):
        from snntoolbox.conversion.utils import normalize_parameters

        # Need to test only once because normalization is independent of
        # input library.
        if 'keras' not in _input_model_and_lib['model_lib'].__name__:
            return

        batch_size = _config.getint('simulation', 'batch_size')
        num_to_test = _config.getint('simulation', 'num_to_test')
        model_parser = _input_model_and_lib['model_lib'].ModelParser(
            _input_model_and_lib['input_model']['model'], _config)
        model_parser.parse()
        parsed_model = model_parser.build_parsed_model()
        normalize_parameters(parsed_model, _config, **_normset)
        score = model_parser.evaluate(batch_size, num_to_test, **_testset)
        target_acc = _input_model_and_lib['target_acc']

        assert round(100 * score[1], 2) >= target_acc


class TestOutputModel:
    """Test building, saving and running the converted SNN model."""

    def test_building(self, _spiking_model_and_sim, _parsed_model, _config):
        spiking_model = _spiking_model_and_sim['spiking_model']
        spiking_model.build(_parsed_model)
        spiking_model.save(_config['paths']['path_wd'],
                           _config['paths']['filename_snn'])
        print(_config['paths']['path_wd'])
        print(_config['paths']['filename_snn'])

    def test_simulating(self, _spiking_model_and_sim, _testset):
        score = _spiking_model_and_sim['spiking_model'].run(**_testset)
        target_acc = float(_spiking_model_and_sim['target_acc'])
        assert round(100 * score, 2) >= target_acc

    def test_brian2(self, _parsed_model, _testset, _config):
        """Needs to be tested separately because no saving function implemented.
        """

        from importlib import import_module
        from snntoolbox.bin.utils import initialize_simulator

        _config.read_dict({'simulation': {'simulator': 'brian2',
                                          'num_to_test': 2},
                           'input': {'poisson_input': True}})
        try:
            initialize_simulator(_config)
        except ImportError:
            return

        target_sim = import_module(
            'snntoolbox.simulation.target_simulators.brian2_target_sim')
        spiking_model = target_sim.SNN(_config)
        spiking_model.build(_parsed_model)
        score = spiking_model.run(**_testset)
        assert round(100 * score, 2) >= 99.00


class TestPipeline:
    """Test complete pipeline for a number of examples."""

    def test_examples(self, _example_filepath):
        from snntoolbox.bin.utils import update_setup, test_full

        config = update_setup(_example_filepath)
        assert test_full(config)[0] >= 0.5
