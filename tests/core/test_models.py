# coding=utf-8

from snntoolbox.core.util import parse, normalize_parameters, evaluate_keras


class TestInputModel:
    """Test loading, parsing and evaluating an input ANN model."""

    def test_loading(self, input_model_and_lib, evalset):
        model_lib = input_model_and_lib['model_lib']
        input_model = input_model_and_lib['input_model']
        score = model_lib.evaluate(input_model['val_fn'], **evalset)
        target_acc = input_model_and_lib['target_acc']
        assert round(100 * score[1], 2) == target_acc

    def test_parsing(self, input_model_and_lib, evalset):
        parsed_model = parse(input_model_and_lib['input_model']['model'])
        score = evaluate_keras(parsed_model, **evalset)
        target_acc = input_model_and_lib['target_acc']
        assert round(100 * score[1], 2) == target_acc

    def test_normalizing(self, input_model_and_lib, normset, evalset, path_wd):
        # Need to test only once because normalization is independent of
        # input library.
        if 'keras' not in input_model_and_lib['model_lib'].__name__:
            assert True
        parsed_model = parse(input_model_and_lib['input_model']['model'])
        normalize_parameters(parsed_model, path=str(path_wd), **normset)
        score = evaluate_keras(parsed_model, **evalset)
        target_acc = input_model_and_lib['target_acc']
        assert round(100 * score[1], 2) == target_acc


class TestOutputModel:
    """Test building, saving and running the converted SNN model."""

    def test_building(self, spiking_model_and_sim, parsed_model, path_wd):
        spiking_model = spiking_model_and_sim['spiking_model']
        spiking_model.build(parsed_model, path_wd=str(path_wd))
        spiking_model.save(str(path_wd),
                           spiking_model_and_sim['target_sim'].__name__)

    def test_simulating(self, spiking_model_and_sim, testset, path_wd,
                        settings):
        score = spiking_model_and_sim['spiking_model'].run(
            path=str(path_wd), settings=settings, **testset)
        target_acc = float(spiking_model_and_sim['target_acc'])
        assert round(100 * score, 2) >= target_acc

    def test_brian2(self, parsed_model, testset, path_wd, settings):
        """Needs to be tested separately because no saving function implemented.
        """

        from importlib import import_module
        from snntoolbox.config import initialize_simulator

        try:
            initialize_simulator('brian2')
        except ImportError:
            return

        target_sim = import_module(
            'snntoolbox.target_simulators.brian2_target_sim')
        settings.update({'simulator': 'brian2', 'num_to_test': 2})
        spiking_model = target_sim.SNN(settings)
        spiking_model.build(parsed_model)
        score = spiking_model.run(path=str(path_wd), settings=settings,
                                  **testset)
        assert round(100 * score, 2) >= 99.00
