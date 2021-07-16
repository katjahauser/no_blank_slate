import collections.abc
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

    def test_get_paths(self):
        experiment_path = "./resources/test_plots/lottery_simplified_experiment/replicate_1"
        expected_paths = utils_tests.generate_expected_paths_for_lottery_replicate(experiment_path, num_levels=2)
        valid_epochs = 2
        evaluator = ConcreteReplicateEvaluator(experiment_path, valid_epochs)

        actual_paths = evaluator.get_paths()

        self.assertDictEqual(expected_paths, actual_paths)

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
            print("Ignoring TestReplicateEvaluator.test_evaluate_experiment_show_and_save to allow automated testing.")

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
            print("Ignoring TestReplicateEvaluator.test_evaluate_experiment_show_but_no_save to allow automated "
                  "testing.")

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
        else:
            print("Ignoring TestSparsityAccuracyOnSingleReplicateEvaluator.test_evaluate_experiment to allow automated "
                  "testing.")


class TestSparsityNeuralPersistenceOnSingleReplicateEvaluator(unittest.TestCase):
    def test_is_subclass_of_ReplicateEvaluator(self):
        valid_num_epochs = 1
        sparsity_neural_persistence_single_replicate_evaluator = \
            experiment.SparsityNeuralPersistenceOnSingleReplicateEvaluator("dummy_path", valid_num_epochs)

        self.assertTrue(issubclass(experiment.SparsityNeuralPersistenceOnSingleReplicateEvaluator,
                                   experiment.ReplicateEvaluator))
        self.assertTrue(isinstance(sparsity_neural_persistence_single_replicate_evaluator,
                                   experiment.ReplicateEvaluator))

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
        else:
            print("Ignoring TestSparsityNeuralPersistenceOnSingleReplicateEvaluator.test_evaluate_experiment to allow "
                  "automated testing.")


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
        else:
            print("Ignoring TestAccuracyNeuralPersistenceOnSingleReplicateEvaluator.test_evaluate_experiment to allow "
                  "automated testing.")


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
        else:
            print("Ignoring TestSparsityAccuracyExperimentEvaluator.test_evaluate_experiment to allow automated "
                  "testing.")


class TestNeuralPersistenceExperimentEvaluator(unittest.TestCase):
    def test_inheritance(self):
        valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_num_epochs = 2
        evaluator = ConcreteNeuralPersistenceExperimentEvaluator(valid_experiment_path, valid_num_epochs)

        self.assertTrue(isinstance(evaluator, experiment.NeuralPersistenceExperimentEvaluator))
        self.assertTrue(issubclass(ConcreteNeuralPersistenceExperimentEvaluator,
                                   experiment.NeuralPersistenceExperimentEvaluator))
        self.assertTrue(issubclass(ConcreteNeuralPersistenceExperimentEvaluator,
                                   experiment.ExperimentEvaluator))
        self.assertTrue(issubclass(ConcreteNeuralPersistenceExperimentEvaluator,
                                   experiment.ReplicateEvaluator))

    def test_load_y_data(self):
        expected_np_level_0, expected_np_level_1 = get_neural_persistences_for_lottery_simplified()
        expected_y_data = [[expected_np_level_0, expected_np_level_1], [expected_np_level_0, expected_np_level_1]]
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = ConcreteNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        paths = evaluator.get_paths()

        evaluator.load_y_data(paths)

        self.assertEqual(expected_y_data, evaluator.y_data)
        for i in range(len(expected_y_data)):
            self.assertDictEqual(expected_y_data[i][0], evaluator.y_data[i][0])
            self.assertDictEqual(expected_y_data[i][1], evaluator.y_data[i][1])

    def test_prepare_data_for_plotting(self):
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = ConcreteNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        expected_x_data = [2, 3, 4]
        expected_y_data = np.asarray(
            [[[0.4472135954999579, 0.4472135954999579],  # layer 1, sparsity level 0, replicates 1 & 2
              [0.6324555320336759, 0.6324555320336759]],  # layer 1, sparsity level 1, "
             [[0.5773502691896258, 0.5773502691896258],  # layer 2, sparsity level 0, "
              [0.8164965809277261, 0.8164965809277261]],  # layer 2, sparsity level 1, "
             [[0.5122819323447919, 0.5122819323447919],  # global, sparsity level 0, "
              [0.724476056480701, 0.724476056480701]]])   # global, sparsity level 1, "

        evaluator.y_data = setup_mock_np_with_nans()
        evaluator.x_data = expected_x_data

        evaluator.prepare_data_for_plotting()

        self.assertEqual(expected_x_data, evaluator.x_data)
        np.testing.assert_array_equal(expected_y_data, evaluator.y_data)

    def test_get_layer_names(self):
        expected_layer_names = ['fc_layers.0.weight', 'fc.weight', 'global']
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = ConcreteNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = setup_mock_np_with_nans()

        actual_layer_names = evaluator.get_layer_names()

        self.assertTrue(isinstance(actual_layer_names, collections.abc.KeysView))
        self.assertEqual(expected_layer_names, list(actual_layer_names))

    def test_get_layer_names_raises_on_reformated_data(self):
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = ConcreteNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = setup_mock_np_with_nans()
        evaluator.reformat_neural_persistences()

        with self.assertRaises(TypeError):
            evaluator.get_layer_names()

    def test_reformat_neural_persistences(self):
        expected_tensor = np.asarray(
            [[[0.4472135954999579, 0.4472135954999579],  # layer 1, sparsity level 0, replicates 1 & 2
              [0.6324555320336759, 0.6324555320336759]],  # layer 1, sparsity level 1, "
             [[0.5773502691896258, 0.5773502691896258],  # layer 2, sparsity level 0, "
              [0.8164965809277261, 0.8164965809277261]],  # layer 2, sparsity level 1, "
             [[0.5122819323447919, 0.5122819323447919],  # global, sparsity level 0, "
              [0.724476056480701, 0.724476056480701]]])   # global, sparsity level 1, "
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = ConcreteNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = setup_mock_np_with_nans()

        evaluator.reformat_neural_persistences()

        np.testing.assert_array_equal(expected_tensor, evaluator.y_data)

    def test_create_tensor_for_reformating(self):
        expected_tensor = np.zeros((3, 2, 2))
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = ConcreteNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = setup_mock_np_with_nans()

        actual_tensor = evaluator.create_tensor_for_reformating()

        np.testing.assert_array_equal(expected_tensor, actual_tensor)

    def test_get_normalized_neural_persistence(self):
        layer_wise_np_dict = {'total_persistence': np.nan, 'total_persistence_normalized': 0.4472135954999579}
        expected_value = 0.4472135954999579
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = ConcreteNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)

        actual_value = evaluator.get_normalized_neural_persistence(layer_wise_np_dict)

        self.assertEqual(expected_value, actual_value)


class ConcreteNeuralPersistenceExperimentEvaluator(experiment.NeuralPersistenceExperimentEvaluator):
    def load_x_data(self, paths):
        raise NotImplementedError("Trying to call load_x_data from mock class "
                                  "ConcreteNeuralPersistenceExperimentEvaluator.")

    def prepare_x_data_for_plotting(self):
        pass

    def prepare_neural_persistences_for_plotting(self):
        pass

    def match_layer_names(self, layer_names):
        pass

    def get_plotter(self):
        raise NotImplementedError("Trying to call get_plotter from mock class "
                                  "ConcreteNeuralPersistenceExperimentEvaluator.")


def setup_mock_np_with_nans():
    # we are only interested in the normalized values -- mistakes in indexing are easier to detect, if we
    # substitute NaN for all entries we are not interested in.
    # these are the neural persistences of get_neural_persistences_for_lottery_simplified() with nans for the
    # non-normalized neural persistences
    neural_persistences_w_nan = \
        [collections.defaultdict(dict, {'fc_layers.0.weight': {'total_persistence': np.nan,
                                                               'total_persistence_normalized': 0.4472135954999579},
                                        'fc.weight': {'total_persistence': np.nan,
                                                      'total_persistence_normalized': 0.5773502691896258},
                                        'global': {'accumulated_total_persistence': np.nan,
                                                   'accumulated_total_persistence_normalized': 0.5122819323447919}}
                                 ),
         collections.defaultdict(dict, {'fc_layers.0.weight': {'total_persistence': np.nan,
                                                               'total_persistence_normalized': 0.6324555320336759},
                                        'fc.weight': {'total_persistence': np.nan,
                                                      'total_persistence_normalized': 0.8164965809277261},
                                        'global': {'accumulated_total_persistence': np.nan,
                                                   'accumulated_total_persistence_normalized': 0.724476056480701}})
         ]
    return [neural_persistences_w_nan, neural_persistences_w_nan]


class TestSparsityNeuralPersistenceExperimentEvaluator(unittest.TestCase):
    def test_inheritance(self):
        valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_num_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(valid_experiment_path, valid_num_epochs)

        self.assertTrue(isinstance(evaluator, experiment.NeuralPersistenceExperimentEvaluator))
        self.assertTrue(issubclass(experiment.SparsityNeuralPersistenceExperimentEvaluator,
                                   experiment.NeuralPersistenceExperimentEvaluator))
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

    def test_prepare_x_data_for_plotting(self):
        expected_sparsities = [1.0, 212959.0/266200.0]
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.load_data()

        evaluator.prepare_x_data_for_plotting()

        self.assertEqual(expected_sparsities, evaluator.x_data)

    def test_prepare_neural_persistences_for_plotting(self):
        # mock_np is two times the neural persistences from get_neural_persistences_for_lottery_simplified() with NaNs
        mock_np = setup_mock_np_with_nans()
        expected_neural_persistence_means = np.asarray([[0.4472135954999579, 0.6324555320336759],
                                                        [0.5773502691896258, 0.8164965809277261],
                                                        [0.5122819323447919, 0.724476056480701]])
        expected_neural_persistence_std = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = mock_np
        evaluator.reformat_neural_persistences()

        evaluator.prepare_neural_persistences_for_plotting()
        actual_means = evaluator.y_data[0]
        actual_std = evaluator.y_data[1]

        np.testing.assert_array_equal(expected_neural_persistence_means, actual_means)
        np.testing.assert_array_equal(expected_neural_persistence_std, actual_std)

    def test_compute_means(self):
        # we use the same values twice in the mock up
        expected_means = np.asarray([[0.4472135954999579, 0.6324555320336759],  # layer 1, sparsity level 0 & 1
                                     [0.5773502691896258, 0.8164965809277261],  # layer 2, sparsity level 0 & 1
                                     [0.5122819323447919, 0.724476056480701]])  # global, sparsity level 0 & 1
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = setup_mock_np_with_nans()
        evaluator.reformat_neural_persistences()

        actual_means = evaluator.compute_means()

        np.testing.assert_array_equal(expected_means, actual_means)

    def test_compute_std_deviations(self):
        expected_std_devs = np.zeros((3, 2))  # since we use the same values twice in the mock up
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = setup_mock_np_with_nans()
        evaluator.reformat_neural_persistences()

        actual_std_devs = evaluator.compute_std_deviations()

        np.testing.assert_array_equal(expected_std_devs, actual_std_devs)

    def test_match_layer_names(self):
        mock_np = setup_mock_np_with_nans()
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
        layer_names = evaluator.get_layer_names()
        evaluator.reformat_neural_persistences()
        evaluator.prepare_neural_persistences_for_plotting()
        assert type(evaluator.y_data[0]) == np.ndarray and type(evaluator.y_data[1]) == np.ndarray

        evaluator.match_layer_names(layer_names)
        actual_means = evaluator.y_data[0]
        actual_std = evaluator.y_data[1]

        self.assertEqual(expected_neural_persistence_means.keys(), actual_means.keys())
        self.assertEqual(expected_neural_persistence_std.keys(), actual_std.keys())
        for key in expected_neural_persistence_means.keys():
            np.testing.assert_array_equal(expected_neural_persistence_means[key], actual_means[key])
            np.testing.assert_array_equal(expected_neural_persistence_std[key], actual_std[key])

    def test_match_layer_names_to_statistic(self):
        expected_means = {'fc_layers.0.weight': [0.4472135954999579, 0.6324555320336759],
                          'fc.weight': [0.5773502691896258, 0.8164965809277261],
                          'global': [0.5122819323447919, 0.724476056480701]}
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = setup_mock_np_with_nans()
        layer_names = evaluator.get_layer_names()
        evaluator.reformat_neural_persistences()
        raw_means = evaluator.compute_means()

        actual_means = evaluator.match_layer_names_to_statistic(layer_names, raw_means)

        self.assertEqual(expected_means.keys(), actual_means.keys())
        for key in expected_means.keys():
            np.testing.assert_array_equal(expected_means[key], actual_means[key])

    def test_get_plotter(self):
        valid_epochs = 2
        valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(valid_experiment_path, valid_epochs)

        plotter = evaluator.get_plotter()

        self.assertTrue(isinstance(plotter, plotters.SparsityNeuralPersistenceExperimentPlotter))

    def test_evaluate_experiment(self):
        # sanity check
        if not show_no_plots_for_automated_tests:
            valid_num_epochs = 2
            valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
            evaluator = experiment.SparsityNeuralPersistenceExperimentEvaluator(valid_experiment_path, valid_num_epochs)

            evaluator.evaluate(True, False)
        else:
            print("Ignoring TestSparsityNeuralPersistenceExperimentEvaluator.test_evaluate_experiment to allow "
                  "automated testing.")


class TestAccuracyNeuralPersistenceExperimentEvaluator(unittest.TestCase):
    def test_inheritance(self):
        valid_num_epochs = 2
        valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        evaluator = experiment.AccuracyNeuralPersistenceExperimentEvaluator(valid_experiment_path, valid_num_epochs)

        self.assertTrue(isinstance(evaluator, experiment.NeuralPersistenceExperimentEvaluator))
        self.assertTrue(issubclass(experiment.AccuracyNeuralPersistenceExperimentEvaluator,
                                   experiment.NeuralPersistenceExperimentEvaluator))
        self.assertTrue(issubclass(experiment.AccuracyNeuralPersistenceExperimentEvaluator,
                                   experiment.ExperimentEvaluator))
        self.assertTrue(issubclass(experiment.AccuracyNeuralPersistenceExperimentEvaluator,
                                   experiment.ReplicateEvaluator))

    def test_load_x_data(self):  # load accuracies
        expected_accuracies = np.asarray([[0.9644, 0.9678], [0.9544, 0.9878]])
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.AccuracyNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        paths = evaluator.get_paths()

        evaluator.load_x_data(paths)

        np.testing.assert_array_equal(expected_accuracies, evaluator.x_data)

    def test_prepare_x_data_for_plotting(self):
        expected_accuracies = np.asarray([[0.9644, 0.9678], [0.9544, 0.9878]])
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.AccuracyNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.load_data()

        evaluator.prepare_x_data_for_plotting()

        np.testing.assert_array_equal(expected_accuracies, evaluator.x_data)

    def test_prepare_y_data_for_plotting(self):
        expected_neural_persistences = np.asarray(
            [[[0.4472135954999579, 0.4472135954999579],  # layer 1, sparsity level 0, replicates 1 & 2
              [0.6324555320336759, 0.6324555320336759]],  # layer 1, sparsity level 1, "
             [[0.5773502691896258, 0.5773502691896258],  # layer 2, sparsity level 0, "
              [0.8164965809277261, 0.8164965809277261]],  # layer 2, sparsity level 1, "
             [[0.5122819323447919, 0.5122819323447919],  # global, sparsity level 0, "
              [0.724476056480701, 0.724476056480701]]])   # global, sparsity level 1, "
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.AccuracyNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = setup_mock_np_with_nans()
        evaluator.reformat_neural_persistences()

        evaluator.prepare_neural_persistences_for_plotting()

        np.testing.assert_array_equal(expected_neural_persistences, evaluator.y_data)

    def test_match_layer_names(self):
        expected_np = {'fc_layers.0.weight': [[0.4472135954999579, 0.4472135954999579],  # layer 1, sparsity level 0, replicates 1 & 2
                                              [0.6324555320336759, 0.6324555320336759]],  # layer 1, sparsity level 1, "
                       'fc.weight': [[0.5773502691896258, 0.5773502691896258],  # layer 2, sparsity level 0, "
                                     [0.8164965809277261, 0.8164965809277261]],  # layer 2, sparsity level 1, "
                       'global': [[0.5122819323447919, 0.5122819323447919],  # global, sparsity level 0, "
                                  [0.724476056480701, 0.724476056480701]]}  # global, sparsity level 1, "
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        evaluator = experiment.AccuracyNeuralPersistenceExperimentEvaluator(experiment_path, valid_epochs)
        evaluator.y_data = setup_mock_np_with_nans()
        layer_names = evaluator.get_layer_names()
        evaluator.reformat_neural_persistences()
        evaluator.prepare_neural_persistences_for_plotting()

        evaluator.match_layer_names(layer_names)

        self.assertEqual(expected_np.keys(), evaluator.y_data.keys())
        for key in expected_np.keys():
            np.testing.assert_array_equal(expected_np[key], evaluator.y_data[key])

    def test_get_plotter(self):
        valid_epochs = 2
        valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        evaluator = experiment.AccuracyNeuralPersistenceExperimentEvaluator(valid_experiment_path, valid_epochs)

        plotter = evaluator.get_plotter()

        self.assertTrue(isinstance(plotter, plotters.AccuracyNeuralPersistenceExperimentPlotter))

    def test_evaluate_experiment(self):
        # sanity check
        if not show_no_plots_for_automated_tests:
            valid_num_epochs = 2
            valid_experiment_path = "./resources/test_plots/lottery_simplified_experiment"
            evaluator = experiment.AccuracyNeuralPersistenceExperimentEvaluator(valid_experiment_path, valid_num_epochs)

            evaluator.evaluate(True, False)
        else:
            print("Ignoring TestSparsityAccuracyExperimentEvaluator.test_evaluate_experiment to allow automated "
                  "testing.")


if __name__ == '__main__':

    unittest.main()
