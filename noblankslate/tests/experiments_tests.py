from collections import OrderedDict
import os.path
import unittest

import matplotlib.axes
import matplotlib.figure
import numpy as np
import torch

from deps.neural_persistence.src.tda import PerLayerCalculation
import src.experiment as experiment
import src.plotters as plotters
import src.utils as utils
import tests.utils_tests as utils_tests


show_plot_off_for_fast_tests = False

show_no_plots_for_automated_tests = True


class TestReplicateEvaluator(unittest.TestCase):
    def test_is_subclass_of_ReplicateEvaluator(self):
        evaluator = ConcreteReplicateEvaluator("dummy_path", 1)

        self.assertTrue(issubclass(ConcreteReplicateEvaluator, experiment.ReplicateEvaluator))
        self.assertTrue(isinstance(evaluator, experiment.ReplicateEvaluator))

    def test_raise_if_no_valid_epoch(self):
        eps_0 = 0
        eps_smaller_0 = -4
        eps_smaller_0_double = -3.5
        eps_greater_0_double = 4.5
        with self.assertRaises(ValueError):
            ConcreteReplicateEvaluator("dummy_path", eps_0)
        with self.assertRaises(ValueError):
            ConcreteReplicateEvaluator("dummy_path", eps_smaller_0)
        with self.assertRaises(ValueError):
            ConcreteReplicateEvaluator("dummy_path", eps_smaller_0_double)
        with self.assertRaises(ValueError):
            ConcreteReplicateEvaluator("dummy_path", eps_greater_0_double)

    def test_load_data(self):
        expected_x_values = [1., 2., 3.]
        expected_y_values = [1., 2., 3.]
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        evaluator = ConcreteReplicateEvaluator(path_to_replicate, valid_num_epochs)
        assert evaluator.x_data == [] and evaluator.y_data == []

        evaluator.load_data()

        self.assertEqual(expected_x_values, evaluator.x_data)
        self.assertEqual(expected_y_values, evaluator.y_data)

    def test_get_plotter(self):
        valid_num_epochs = 2
        evaluator = ConcreteReplicateEvaluator("dummy_path", valid_num_epochs)

        plotter = evaluator.get_plotter()

        self.assertTrue(isinstance(plotter, MockPlotter))

    def test_generate_plot(self):
        valid_num_epochs = 2
        evaluator = ConcreteReplicateEvaluator("dummy_path", valid_num_epochs)
        plotter = evaluator.get_plotter()

        evaluator.generate_plot(plotter)

        # the two tests below implicitly test that plotter.make_plot was called -- plotter.axis and plotter.figure are
        # None otherwise
        self.assertTrue(isinstance(plotter.axis, matplotlib.axes.SubplotBase))
        self.assertTrue(isinstance(plotter.figure, matplotlib.figure.Figure))

    def test_evaluate_experiment_show_and_save(self):
        if not show_no_plots_for_automated_tests:
            valid_num_epochs = 2
            path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
            assert os.path.exists(path_to_replicate)
            evaluator = ConcreteReplicateEvaluator(path_to_replicate, valid_num_epochs)
            target_file = "./resources/test_plots/lottery_simplified/plots/ConcreteReplicateEvaluator_plot.png"
            remove_target_file_if_exists(target_file)

            evaluator.evaluate(show_plot=True, save_plot=True)

            self.assertTrue(os.path.exists(target_file))
        else:
            print("Ignoring TestSparsityAccuracyOnSingleReplicateEvaluator.test_evaluate_experiment_show_and_save to "
                  "allow automated testing.")

    def test_evaluate_experiment_show_but_no_save(self):
        if not show_no_plots_for_automated_tests:
            valid_num_epochs = 2
            path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
            assert os.path.exists(path_to_replicate)
            evaluator = ConcreteReplicateEvaluator(path_to_replicate, valid_num_epochs)
            target_file = "./resources/test_plots/lottery_simplified/plots/ConcreteReplicateEvaluator_plot.png"
            remove_target_file_if_exists(target_file)

            evaluator.evaluate(show_plot=True, save_plot=False)

            self.assertFalse(os.path.exists(target_file))
        else:
            print("Ignoring TestSparsityAccuracyOnSingleReplicateEvaluator.test_evaluate_experiment_show_but_no_save "
                  "to allow automated testing.")

    def test_evaluate_experiment_no_show_but_save(self):
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert os.path.exists(path_to_replicate)
        evaluator = ConcreteReplicateEvaluator(path_to_replicate, valid_num_epochs)
        target_file = "./resources/test_plots/lottery_simplified/plots/ConcreteReplicateEvaluator_plot.png"
        remove_target_file_if_exists(target_file)

        evaluator.evaluate(show_plot=False, save_plot=True)

        self.assertTrue(os.path.exists(target_file))

    def test_evaluate_experiment_no_show_no_save(self):
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert os.path.exists(path_to_replicate)
        evaluator = ConcreteReplicateEvaluator(path_to_replicate, valid_num_epochs)
        target_file = "./resources/test_plots/lottery_simplified/plots/ConcreteReplicateEvaluator_plot.png"
        remove_target_file_if_exists(target_file)

        evaluator.evaluate(show_plot=False, save_plot=False)

        self.assertFalse(os.path.exists(target_file))


class ConcreteReplicateEvaluator(experiment.ReplicateEvaluator):
    # mock implementation for testing non-abstract methods in ReplicateEvaluator.
    def get_paths(self):
        return ""

    def load_x_data(self, paths):
        self.x_data = [1., 2., 3.]

    def load_y_data(self, paths):
        self.y_data = [1., 2., 3.]

    def prepare_data_for_plotting(self):
        pass

    def get_plotter(self):
        return MockPlotter()


class MockPlotter(plotters.PlotterBaseClass):
    title = "ConcreteReplicateEvaluator title"
    x_label = "ConcreteReplicateEvaluator x-label"
    y_label = "ConcreteReplicateEvaluator y-label"
    save_file_name = "ConcreteReplicateEvaluator_plot.png"

    def plot_data(self, axis, x_values, y_values):
        axis.plot(x_values, y_values)


def check_simplified_lottery_experiment_replicate_exists(path_to_replicate):
    if path_to_replicate[-1] != "/":
        path_to_replicate = path_to_replicate + "/"

    assert os.path.exists(path_to_replicate + "level_0/main/model_ep0_it0.pth")
    assert os.path.exists(path_to_replicate + "level_0/main/model_ep2_it0.pth")
    assert os.path.exists(path_to_replicate + "level_1/main/model_ep0_it0.pth")
    assert os.path.exists(path_to_replicate + "level_1/main/model_ep2_it0.pth")
    assert os.path.exists(path_to_replicate + "level_0/main/mask.pth")
    assert os.path.exists(path_to_replicate + "level_1/main/mask.pth")
    assert os.path.exists(path_to_replicate + "level_0/main/logger")
    assert os.path.exists(path_to_replicate + "level_1/main/logger")
    assert os.path.exists(path_to_replicate + "level_0/main/sparsity_report.json")
    assert os.path.exists(path_to_replicate + "level_1/main/sparsity_report.json")

    return True


def remove_target_file_if_exists(target_file):
    if os.path.exists(target_file):
        os.remove(target_file)
    assert not os.path.exists(target_file)


class TestSparsityAccuracyOnSingleReplicateEvaluator(unittest.TestCase):
    def test_is_subclass_of_ReplicateEvaluator(self):
        sparsity_accuracy_single_replicate_evaluator = \
            experiment.SparsityAccuracyOnSingleReplicateEvaluator("dummy_path", 1)
        self.assertTrue(issubclass(experiment.SparsityAccuracyOnSingleReplicateEvaluator,
                                   experiment.ReplicateEvaluator))
        self.assertTrue(isinstance(sparsity_accuracy_single_replicate_evaluator, experiment.ReplicateEvaluator))

    def test_get_paths(self):
        experiment_path = "./resources/test_plots/lottery_simplified_experiment/replicate_1"
        expected_paths = utils_tests.generate_expected_paths_for_lottery_replicate(experiment_path, num_levels=2)
        valid_epochs = 2
        evaluator = experiment.SparsityAccuracyOnSingleReplicateEvaluator(experiment_path, valid_epochs)

        actual_paths = evaluator.get_paths()

        self.assertDictEqual(expected_paths, actual_paths)

    def test_load_x_data(self):  # loads sparsities
        expected_sparsities = [1.0, 212959.0/266200.0]
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        sparsity_accuracy_single_replicate_evaluator = experiment.SparsityAccuracyOnSingleReplicateEvaluator(
            path_to_replicate, valid_num_epochs)
        paths = sparsity_accuracy_single_replicate_evaluator.get_paths()

        sparsity_accuracy_single_replicate_evaluator.load_x_data(paths)

        self.assertEqual(expected_sparsities, sparsity_accuracy_single_replicate_evaluator.x_data)

    def test_load_y_data(self):  # loads accuracies
        expected_accuracies = [0.9644, 0.9678]
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        sparsity_accuracy_single_replicate_evaluator = experiment.SparsityAccuracyOnSingleReplicateEvaluator(
            path_to_replicate, valid_num_epochs)
        paths = sparsity_accuracy_single_replicate_evaluator.get_paths()

        sparsity_accuracy_single_replicate_evaluator.load_y_data(paths)

        self.assertEqual(expected_accuracies, sparsity_accuracy_single_replicate_evaluator.y_data)

    def test_prepare_data_for_plotting(self):
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        sparsity_accuracy_single_replicate_evaluator = \
            experiment.SparsityAccuracyOnSingleReplicateEvaluator(path_to_replicate, valid_num_epochs)
        sparsity_accuracy_single_replicate_evaluator.load_data()
        expected_sparsity = sparsity_accuracy_single_replicate_evaluator.x_data
        expected_accuracy = sparsity_accuracy_single_replicate_evaluator.y_data

        sparsity_accuracy_single_replicate_evaluator.prepare_data_for_plotting()

        self.assertEqual(expected_sparsity, sparsity_accuracy_single_replicate_evaluator.x_data)
        self.assertEqual(expected_accuracy, sparsity_accuracy_single_replicate_evaluator.y_data)

    def test_get_plotter(self):
        valid_num_epochs = 2
        sparsity_accuracy_single_replicate_evaluator = experiment.SparsityAccuracyOnSingleReplicateEvaluator(
            "dummy_path", valid_num_epochs)

        plotter = sparsity_accuracy_single_replicate_evaluator.get_plotter()

        self.assertTrue(isinstance(plotter, plotters.SparsityAccuracyReplicatePlotter))

    def test_evaluate_experiment(self):
        # sanity check
        if not show_no_plots_for_automated_tests:
            valid_num_epochs = 2
            path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
            assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
            sparsity_accuracy_single_replicate_evaluator = experiment.SparsityAccuracyOnSingleReplicateEvaluator(
                path_to_replicate, valid_num_epochs)

            sparsity_accuracy_single_replicate_evaluator.evaluate(True, False)


class TestSparsityNeuralPersistenceOnSingleReplicateEvaluator(unittest.TestCase):
    def test_is_subclass_of_ReplicateEvaluator(self):
        valid_num_epochs = 1
        sparsity_neural_persistence_single_replicate_evaluator = \
            experiment.SparsityNeuralPersistenceOnSingleReplicateEvaluator("dummy_path", valid_num_epochs)

        self.assertTrue(issubclass(experiment.SparsityNeuralPersistenceOnSingleReplicateEvaluator,
                                   experiment.ReplicateEvaluator))
        self.assertTrue(isinstance(sparsity_neural_persistence_single_replicate_evaluator,
                                   experiment.ReplicateEvaluator))

    def test_get_paths(self):
        experiment_path = "./resources/test_plots/lottery_simplified_experiment/replicate_1"
        expected_paths = utils_tests.generate_expected_paths_for_lottery_replicate(experiment_path, num_levels=2)
        valid_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceOnSingleReplicateEvaluator(experiment_path, valid_epochs)

        actual_paths = evaluator.get_paths()

        self.assertDictEqual(expected_paths, actual_paths)

    def test_load_x_data(self):  # loads sparsities
        expected_sparsities = [1.0, 212959.0/266200.0]
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        sparsity_np_single_replicate_evaluator = experiment.SparsityNeuralPersistenceOnSingleReplicateEvaluator(
            path_to_replicate, valid_num_epochs)
        paths = sparsity_np_single_replicate_evaluator.get_paths()

        sparsity_np_single_replicate_evaluator.load_x_data(paths)

        self.assertEqual(expected_sparsities, sparsity_np_single_replicate_evaluator.x_data)

    def test_load_y_data(self):  # loads neural persistences
        expected_np_level_0, expected_np_level_1 = get_neural_persistences_for_lottery_simplified()
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        sparsity_np_single_replicate_evaluator = experiment.SparsityNeuralPersistenceOnSingleReplicateEvaluator(
            path_to_replicate, valid_num_epochs)
        paths = sparsity_np_single_replicate_evaluator.get_paths()

        sparsity_np_single_replicate_evaluator.load_y_data(paths)

        self.assertDictEqual(expected_np_level_0, sparsity_np_single_replicate_evaluator.y_data[0])
        self.assertDictEqual(expected_np_level_1, sparsity_np_single_replicate_evaluator.y_data[1])

    def test_prepare_data_for_plotting(self):
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        sparsity_np_single_replicate_evaluator = experiment.SparsityNeuralPersistenceOnSingleReplicateEvaluator(
            path_to_replicate, valid_num_epochs)
        sparsity_np_single_replicate_evaluator.load_data()
        expected_sparsity = sparsity_np_single_replicate_evaluator.x_data
        expected_neural_persistence = utils.prepare_neural_persistence_for_plotting(
            sparsity_np_single_replicate_evaluator.y_data)

        sparsity_np_single_replicate_evaluator.prepare_data_for_plotting()

        self.assertEqual(expected_sparsity, sparsity_np_single_replicate_evaluator.x_data)
        self.assertEqual(expected_neural_persistence, sparsity_np_single_replicate_evaluator.y_data)

    def test_get_plotter(self):
        valid_num_epochs = 2
        sparsity_np_single_replicate_evaluator = experiment.SparsityNeuralPersistenceOnSingleReplicateEvaluator(
                "dummy_path", valid_num_epochs)

        plotter = sparsity_np_single_replicate_evaluator.get_plotter()

        self.assertTrue(isinstance(plotter, plotters.SparsityNeuralPersistenceReplicatePlotter))

    def test_evaluate_experiment(self):
        # sanity check
        if not show_no_plots_for_automated_tests:
            valid_num_epochs = 2
            path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
            assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
            sparsity_np_single_replicate_evaluator = experiment.SparsityNeuralPersistenceOnSingleReplicateEvaluator(
                    path_to_replicate, valid_num_epochs)

            sparsity_np_single_replicate_evaluator.evaluate(True, False)


def get_neural_persistences_for_lottery_simplified():
    expected_dict_level_0 = OrderedDict([('fc_layers.0.weight', torch.tensor([[-1., -1., -1., -1.],
                                                                              [-1., -1., -1., -1.]]).numpy()),
                                         ('fc.weight', torch.tensor([[-1., -1.],
                                                                     [-1., -1.]]).numpy())])
    expected_dict_level_1 = OrderedDict([('fc_layers.0.weight', torch.tensor([[-1., 0., -1., 0.],
                                                                              [0., -1., -1., 0.]]).numpy()),
                                         ('fc.weight', torch.tensor([[0., 0.],
                                                                     [-1., -1.]]).numpy())])
    per_layer_calc = PerLayerCalculation()
    expected_neural_persistences_level_0 = per_layer_calc(expected_dict_level_0)
    expected_neural_persistences_level_1 = per_layer_calc(expected_dict_level_1)
    return expected_neural_persistences_level_0, expected_neural_persistences_level_1


class TestAccuracyNeuralPersistenceOnSingleReplicateEvaluator(unittest.TestCase):
    def test_is_subclass_of_ReplicateEvaluator(self):
        valid_num_epochs = 1
        accuracy_neural_persistence_single_replicate_evaluator = \
            experiment.AccuracyNeuralPersistenceOnSingleReplicateEvaluator("dummy_path", valid_num_epochs)

        self.assertTrue(issubclass(experiment.AccuracyNeuralPersistenceOnSingleReplicateEvaluator,
                                   experiment.ReplicateEvaluator))
        self.assertTrue(isinstance(accuracy_neural_persistence_single_replicate_evaluator,
                                   experiment.ReplicateEvaluator))

    def test_get_paths(self):
        experiment_path = "./resources/test_plots/lottery_simplified_experiment/replicate_1"
        expected_paths = utils_tests.generate_expected_paths_for_lottery_replicate(experiment_path, num_levels=2)
        valid_epochs = 2
        evaluator = experiment.AccuracyNeuralPersistenceOnSingleReplicateEvaluator(experiment_path, valid_epochs)

        actual_paths = evaluator.get_paths()

        self.assertDictEqual(expected_paths, actual_paths)

    def test_load_x_data(self):  # loads accuracies
        expected_accuracies = [0.9644, 0.9678]
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        accuracy_np_single_replicate_evaluator = experiment.AccuracyNeuralPersistenceOnSingleReplicateEvaluator(
            path_to_replicate, valid_num_epochs)
        paths = accuracy_np_single_replicate_evaluator.get_paths()

        accuracy_np_single_replicate_evaluator.load_x_data(paths)

        self.assertEqual(expected_accuracies, accuracy_np_single_replicate_evaluator.x_data)

    def test_load_y_data(self):  # loads neural persistences
        expected_np_level_0, expected_np_level_1 = get_neural_persistences_for_lottery_simplified()
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        accuracy_np_single_replicate_evaluator = experiment.AccuracyNeuralPersistenceOnSingleReplicateEvaluator(
            path_to_replicate, valid_num_epochs)
        paths = accuracy_np_single_replicate_evaluator.get_paths()

        accuracy_np_single_replicate_evaluator.load_y_data(paths)

        self.assertDictEqual(expected_np_level_0, accuracy_np_single_replicate_evaluator.y_data[0])
        self.assertDictEqual(expected_np_level_1, accuracy_np_single_replicate_evaluator.y_data[1])

    def test_prepare_data_for_plotting(self):
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        accuracy_np_single_replicate_evaluator = experiment.AccuracyNeuralPersistenceOnSingleReplicateEvaluator(
            path_to_replicate, valid_num_epochs)
        accuracy_np_single_replicate_evaluator.load_data()
        expected_accuracy = accuracy_np_single_replicate_evaluator.x_data
        expected_neural_persistence = utils.prepare_neural_persistence_for_plotting(
            accuracy_np_single_replicate_evaluator.y_data)

        accuracy_np_single_replicate_evaluator.prepare_data_for_plotting()

        self.assertEqual(expected_accuracy, accuracy_np_single_replicate_evaluator.x_data)
        self.assertEqual(expected_neural_persistence, accuracy_np_single_replicate_evaluator.y_data)

    def test_get_plotter(self):
        valid_num_epochs = 2
        accuracy_np_single_replicate_evaluator = experiment.AccuracyNeuralPersistenceOnSingleReplicateEvaluator(
                "dummy_path", valid_num_epochs)

        plotter = accuracy_np_single_replicate_evaluator.get_plotter()

        self.assertTrue(isinstance(plotter, plotters.AccuracyNeuralPersistenceReplicatePlotter))

    def test_evaluate_experiment(self):
        # sanity check
        if not show_no_plots_for_automated_tests:
            valid_num_epochs = 2
            path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
            assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
            accuracy_np_single_replicate_evaluator = experiment.AccuracyNeuralPersistenceOnSingleReplicateEvaluator(
                    path_to_replicate, valid_num_epochs)

            accuracy_np_single_replicate_evaluator.evaluate(True, False)


class TestExperimentEvaluator(unittest.TestCase):
    def test_inheritance(self):
        valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_num_epochs = 2
        evaluator = ConcreteExperimentEvaluator(valid_experiment_path, valid_num_epochs)

        self.assertTrue(isinstance(evaluator, experiment.ExperimentEvaluator))
        self.assertTrue(issubclass(ConcreteExperimentEvaluator, experiment.ExperimentEvaluator))
        self.assertTrue(issubclass(ConcreteExperimentEvaluator, experiment.ReplicateEvaluator))

    def test_initialization(self):
        valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_num_epochs = 2
        expected_num_replicates = 2

        evaluator = ConcreteExperimentEvaluator(valid_experiment_path, valid_num_epochs)

        self.assertEqual(expected_num_replicates, evaluator.num_replicates)

    def test_get_num_replicates(self):
        valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_num_epochs = 2
        expected_num_replicates = 2
        evaluator = ConcreteExperimentEvaluator(valid_experiment_path, valid_num_epochs)

        actual_num_replicates = evaluator.get_num_replicates()

        self.assertEqual(expected_num_replicates, actual_num_replicates)

    def test_get_paths(self):
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        expected_paths = utils_tests.generate_expected_paths_for_lottery_experiment(experiment_path, num_replicates=2,
                                                                                    num_levels=2)
        valid_epochs = 2
        evaluator = ConcreteExperimentEvaluator(experiment_path, valid_epochs)

        actual_paths = evaluator.get_paths()

        self.assertDictEqual(expected_paths, actual_paths)


class ConcreteExperimentEvaluator(experiment.ExperimentEvaluator):
    def load_x_data(self, paths):
        pass

    def load_y_data(self, paths):
        pass

    def prepare_data_for_plotting(self):
        pass

    def get_plotter(self):
        pass


class TestSparsityAccuracyExperimentEvaluator(unittest.TestCase):
    def test_inheritance(self):
        valid_num_epochs = 2
        valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        sparsity_accuracy_experiment_evaluator = \
            experiment.SparsityAccuracyExperimentEvaluator(valid_experiment_path, valid_num_epochs)

        self.assertTrue(isinstance(sparsity_accuracy_experiment_evaluator,
                                   experiment.ExperimentEvaluator))
        self.assertTrue(issubclass(experiment.SparsityAccuracyExperimentEvaluator,
                                   experiment.ExperimentEvaluator))
        self.assertTrue(issubclass(experiment.SparsityAccuracyExperimentEvaluator,
                                   experiment.ReplicateEvaluator))

    def test_load_x_data(self):  # load sparsities
        expected_sparsities = [1.0, 212959.0/266200.0]
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityAccuracyExperimentEvaluator(experiment_path, valid_epochs)
        paths = evaluator.get_paths()

        evaluator.load_x_data(paths)

        self.assertEqual(expected_sparsities, evaluator.x_data)

    def test_load_y_data(self):  # load accuracies
        expected_accuracies = np.asarray([[0.9644, 0.9678], [0.9544, 0.9878]])
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityAccuracyExperimentEvaluator(experiment_path, valid_epochs)
        paths = evaluator.get_paths()

        evaluator.load_y_data(paths)

        np.testing.assert_array_equal(expected_accuracies, evaluator.y_data)

    def test_prepare_data_for_plotting(self):
        accuracies = np.asarray([[0.9644, 0.9678], [0.9544, 0.9878]])
        expected_means = [0.9594, 0.9778]
        expected_std_devs = [0.005, 0.01]
        expected_sparsities = [1.0, 212959.0/266200.0]
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityAccuracyExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.load_data()
        assert evaluator.x_data == expected_sparsities and (evaluator.y_data == accuracies).all()

        evaluator.prepare_data_for_plotting()

        self.assertEqual(expected_sparsities, evaluator.x_data)
        self.assertEqual(expected_means, evaluator.y_data[0])
        for i in range(len(expected_std_devs)):
            self.assertAlmostEqual(expected_std_devs[i], evaluator.y_data[1][i])

    def test_get_plotter(self):
        valid_epochs = 2
        valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        evaluator = experiment.SparsityAccuracyExperimentEvaluator(valid_experiment_path, valid_epochs)

        plotter = evaluator.get_plotter()

        self.assertTrue(isinstance(plotter, plotters.SparsityAccuracyExperimentPlotter))

    def test_evaluate_experiment(self):
        # sanity check
        if not show_no_plots_for_automated_tests:
            valid_num_epochs = 2
            valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
            evaluator = experiment.SparsityAccuracyExperimentEvaluator(valid_experiment_path, valid_num_epochs)

            evaluator.evaluate(True, False)


class TestLoadingSparsitiesForExperiment(unittest.TestCase):
    def test_load_sparsities_for_experiment(self):
        expected_sparsities = [1.0, 212959.0/266200.0]
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        paths = utils.get_paths_from_experiment(experiment_path, "lottery", valid_epochs)

        actual_sparsities = experiment.load_sparsities_for_experiment(paths)

        self.assertEqual(expected_sparsities, actual_sparsities)

    def test_load_sparsities_on_unequal_sparsities_raises(self):
        experiment_path = "./resources/test_get_paths_from_experiment/lottery_simplified_experiment_unequal_sparsities"
        valid_epochs = 2
        paths = utils.get_paths_from_experiment(experiment_path, "lottery", valid_epochs)

        with self.assertRaises(AssertionError):
            experiment.load_sparsities_for_experiment(paths)

    def test_assert_sparsities_equal_raises_on_unequal_sparsities(self):
        sparsities1 = [1., .2]
        sparsities2 = [1., .1]

        with self.assertRaises(AssertionError):
            experiment.assert_sparsities_equal(sparsities1, sparsities2, "key1", "key2")

    def test_assert_sparsities_equal_does_not_raise_on_equal_sparsities(self):
        sparsities1 = [1., .1]
        sparsities2 = [1., .1]

        should_be_none = experiment.assert_sparsities_equal(sparsities1, sparsities2, "key1", "key2")

        self.assertIsNone(should_be_none)


class TestSparsityNeuralPersistenceExperimentEvaluator(unittest.TestCase):
    def test_inheritance(self):
        valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_num_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(valid_experiment_path, valid_num_epochs)

        self.assertTrue(isinstance(evaluator, experiment.ExperimentEvaluator))
        self.assertTrue(issubclass(experiment.SparsityNeuralPersistenceExperimentEvaluator,
                                   experiment.ExperimentEvaluator))
        self.assertTrue(issubclass(experiment.SparsityNeuralPersistenceExperimentEvaluator,
                                   experiment.ReplicateEvaluator))

    def test_load_x_data(self):  # load sparsities
        expected_sparsities = [1.0, 212959.0/266200.0]
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        paths = evaluator.get_paths()

        evaluator.load_x_data(paths)

        self.assertEqual(expected_sparsities, evaluator.x_data)

    def test_load_y_data(self):
        expected_np_level_0, expected_np_level_1 = get_neural_persistences_for_lottery_simplified()
        expected_y_data = [[expected_np_level_0, expected_np_level_1], [expected_np_level_0, expected_np_level_1]]
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        paths = evaluator.get_paths()

        evaluator.load_y_data(paths)

        self.assertEqual(expected_y_data, evaluator.y_data)
        for i in range(len(expected_y_data)):
            self.assertDictEqual(expected_y_data[i][0], evaluator.y_data[i][0])
            self.assertDictEqual(expected_y_data[i][1], evaluator.y_data[i][1])

    def test_prepare_data_for_plotting(self):
        # we are only interested in the normalized values -- mistakes in indexing are easier to detect, if we
        # substitute NaN for all entries we are not interested in.
        # mock_np is two times the neural persistences from get_neural_persistences_for_lottery_simplified() with NaNs
        mock_np = self.setup_mock_np_with_nans()
        expected_neural_persistence_means = {'fc_layers.0.weight': [0.4472135954999579, 0.6324555320336759],
                                             'fc.weight': [0.5773502691896258, 0.8164965809277261],
                                             'global': [0.5122819323447919, 0.724476056480701]}
        expected_neural_persistence_std = {'fc_layers.0.weight': [0.0, 0.0],
                                             'fc.weight': [0.0, 0.0],
                                             'global': [0.0, 0.0]}
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = mock_np

        evaluator.prepare_data_for_plotting()

        self.assertDictEqual(expected_neural_persistence_means, evaluator.y_data[0])
        self.assertDictEqual(expected_neural_persistence_std, evaluator.y_data[1])

    @staticmethod
    def setup_mock_np_with_nans():
        # these are the neural persistences of get_neural_persistences_for_lottery_simplified() with nans for the
        # non-normalized neural persistences
        neural_persistences_w_nan = [{'fc_layers.0.weight': {'total_persistence': np.nan,
                                                             'total_persistence_normalized': 0.4472135954999579},
                                      'fc.weight': {'total_persistence': np.nan,
                                                    'total_persistence_normalized': 0.5773502691896258},
                                      'global': {'accumulated_total_persistence': np.nan,
                                                 'accumulated_total_persistence_normalized': 0.5122819323447919}},
                                     {'fc_layers.0.weight': {'total_persistence': np.nan,
                                                             'total_persistence_normalized': 0.6324555320336759},
                                      'fc.weight': {'total_persistence': np.nan,
                                                    'total_persistence_normalized': 0.8164965809277261},
                                      'global': {'accumulated_total_persistence': np.nan,
                                                 'accumulated_total_persistence_normalized': 0.724476056480701}},
                                     ]
        return [neural_persistences_w_nan, neural_persistences_w_nan]

    def test_reformat_neural_persistences(self):
        expected_tensor = np.asarray([[[0.4472135954999579, 0.4472135954999579],   # layer 1, sparsity level 0, replicates 1 & 2
                                       [0.6324555320336759, 0.6324555320336759]],  # layer 1, sparsity level 1, "
                                      [[0.5773502691896258, 0.5773502691896258],   # layer 2, sparsity level 0, "
                                       [0.8164965809277261, 0.8164965809277261]],  # layer 2, sparsity level 1, "
                                      [[0.5122819323447919, 0.5122819323447919],   # layer 3, sparsity level 0, "
                                       [0.724476056480701, 0.724476056480701]]])   # layer 3, sparsity level 1, "
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = self.setup_mock_np_with_nans()
        paths = evaluator.get_paths()
        evaluator.load_x_data(paths)

        evaluator.reformat_neural_persistences()

        np.testing.assert_array_equal(expected_tensor, evaluator.y_data)

    def test_create_tensor_for_reformating(self):
        expected_tensor = np.zeros((3, 2, 2))
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = self.setup_mock_np_with_nans()
        paths = evaluator.get_paths()
        evaluator.load_x_data(paths)

        actual_tensor = evaluator.create_tensor_for_reformating()

        np.testing.assert_array_equal(expected_tensor, actual_tensor)

    def test_get_normalized_neural_persistence(self):
        self.assertTrue(False) # todo


class TestSparsityNeuralPersistenceExperiment(unittest.TestCase):
    def test_sparsity_np(self):
        expected_sparsities = [1.0, 212959.0/266200.0]

        expected_dict_level_0 = OrderedDict([('fc_layers.0.weight', torch.tensor([[-1., -1., -1., -1.],
                                                                                  [-1., -1., -1., -1.]]).numpy()),
                                             ('fc.weight', torch.tensor([[-1., -1.], [-1., -1.]]).numpy())])
        expected_dict_level_1 = OrderedDict([('fc_layers.0.weight', torch.tensor([[-1., 0., -1., 0.],
                                                                                  [0., -1., -1., 0.]]).numpy()),
                                             ('fc.weight', torch.tensor([[0., 0.], [-1., -1.]]).numpy())])
        per_layer_calc = PerLayerCalculation()

        expected_np_lvl_0 = per_layer_calc(expected_dict_level_0)
        expected_np_lvl_1 = per_layer_calc(expected_dict_level_1)
        expected_mean_np_lvl_0 = {layer_key: expected_np_lvl_0[layer_key][persistence_key] for layer_key in
                                  expected_np_lvl_0.keys()
                                  for persistence_key in expected_np_lvl_0[layer_key].keys()
                                  if "normalized" in persistence_key}
        expected_mean_np_lvl_1 = {layer_key: expected_np_lvl_1[layer_key][persistence_key] for layer_key in
                                  expected_np_lvl_1.keys()
                                  for persistence_key in expected_np_lvl_1[layer_key].keys()
                                  if "normalized" in persistence_key}

        expected_std_np_lvl_0 = {key: 0.0 for key in expected_np_lvl_0.keys()}
        expected_std_np_lvl_1 = {key: 0.0 for key in expected_np_lvl_1.keys()}

        sparsities, mean_nps, std_nps = experiment.sparsity_neural_persistence_plot_experiment(
            "./resources/test_plots/lottery_simplified_experiment/", 2, show_plot=show_plot_off_for_fast_tests, save_plot=False)

        self.assertEqual(expected_sparsities, sparsities)
        self.assertEqual(expected_mean_np_lvl_0, mean_nps[0])
        self.assertEqual(expected_mean_np_lvl_1, mean_nps[1])
        self.assertEqual(expected_std_np_lvl_0, std_nps[0])
        self.assertEqual(expected_std_np_lvl_1, std_nps[1])

    def test_equal_sparsity_lengths(self):
        with self.assertRaises(AssertionError):
            experiment.sparsity_neural_persistence_plot_experiment(
                "./resources/test_get_paths_from_experiment/lottery_simplified_experiment_unequal_sparsities", 2,
                show_plot=show_plot_off_for_fast_tests, save_plot=False)

    def test_save_plot(self):
        if os.path.isfile("./resources/test_plots/lottery_simplified_experiment/plots/sparsity_neural_persistence_experiment.png"):
            os.remove("./resources/test_plots/lottery_simplified_experiment/plots/sparsity_neural_persistence_experiment.png")

        assert not os.path.exists(
            "./resources/test_plots/lottery_simplified_experiment/plots/sparsity_neural_persistence_experiment.png")

        _, _, _ = experiment.sparsity_neural_persistence_plot_experiment(
            "./resources/test_plots/lottery_simplified_experiment/", 2, show_plot=show_plot_off_for_fast_tests, save_plot=True)
        self.assertTrue(os.path.isfile(
            "./resources/test_plots/lottery_simplified_experiment/plots/sparsity_neural_persistence_experiment.png"))


class TestAccuracyNPPlotExperiment(unittest.TestCase):
    def test_accuracy_NP_experiment(self):
        expected_accuracies = [0.9644, 0.9678, 0.9544, 0.9878]

        expected_dict_level_0 = OrderedDict([('fc_layers.0.weight', torch.tensor([[-1., -1., -1., -1.],
                                                                                  [-1., -1., -1., -1.]]).numpy()),
                                             ('fc.weight', torch.tensor([[-1., -1.], [-1., -1.]]).numpy())])
        expected_dict_level_1 = OrderedDict([('fc_layers.0.weight', torch.tensor([[-1., 0., -1., 0.],
                                                                                  [0., -1., -1., 0.]]).numpy()),
                                             ('fc.weight', torch.tensor([[0., 0.], [-1., -1.]]).numpy())])
        per_layer_calc = PerLayerCalculation()

        expected_np_lvl_0 = per_layer_calc(expected_dict_level_0)
        expected_np_lvl_1 = per_layer_calc(expected_dict_level_1)
        expected_neural_persistences = [expected_np_lvl_0, expected_np_lvl_1, expected_np_lvl_0, expected_np_lvl_1]

        accuracies, neural_persistences = experiment.accuracy_neural_persistence_plot_experiment(
            "./resources/test_plots/lottery_simplified_experiment/", 2, show_plot=show_plot_off_for_fast_tests, save_plot=False)

        self.assertEqual(expected_accuracies, accuracies)
        for i in range(len(expected_neural_persistences)):
            self.assertDictEqual(expected_neural_persistences[i], neural_persistences[i])

    def test_accuracy_NP_plot(self):
        if os.path.isfile("./resources/test_plots/lottery_simplified_experiment/plots/accuracy_neural_persistence_experiment.png"):
            os.remove("./resources/test_plots/lottery_simplified_experiment/plots/accuracy_neural_persistence_experiment.png")

        assert not os.path.exists(
            "./resources/test_plots/lottery_simplified_experiment/plots/accuracy_neural_persistence_experiment.png")

        _, _ = experiment.accuracy_neural_persistence_plot_experiment(
            "./resources/test_plots/lottery_simplified_experiment/", 2, show_plot=show_plot_off_for_fast_tests, save_plot=True)
        self.assertTrue(os.path.isfile(
            "./resources/test_plots/lottery_simplified_experiment/plots/accuracy_neural_persistence_experiment.png"))


if __name__ == '__main__':

    unittest.main()
