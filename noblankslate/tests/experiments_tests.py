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


show_plot_off_for_fast_tests = False

show_no_plots_for_automated_tests = True


class TestSingleReplicateHandler(unittest.TestCase):
    def test_is_subclass_of_SingleReplicateHandler(self):
        handler = ConcreteSingleReplicateHandler("dummy_path", 1)

        self.assertTrue(issubclass(ConcreteSingleReplicateHandler, experiment.SingleReplicateHandler))
        self.assertTrue(isinstance(handler, experiment.SingleReplicateHandler))

    def test_raise_if_no_valid_epoch(self):
        eps_0 = 0
        eps_smaller_0 = -4
        eps_smaller_0_double = -3.5
        eps_greater_0_double = 4.5
        with self.assertRaises(ValueError):
            handler = ConcreteSingleReplicateHandler("dummy_path", eps_0)
        with self.assertRaises(ValueError):
            handler = ConcreteSingleReplicateHandler("dummy_path", eps_smaller_0)
        with self.assertRaises(ValueError):
            handler = ConcreteSingleReplicateHandler("dummy_path", eps_smaller_0_double)
        with self.assertRaises(ValueError):
            handler = ConcreteSingleReplicateHandler("dummy_path", eps_greater_0_double)

    def test_load_data(self):
        expected_x_values = [1., 2., 3.]
        expected_y_values = [1., 2., 3.]
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        handler = ConcreteSingleReplicateHandler(path_to_replicate, valid_num_epochs)
        assert handler.x_data == [] and handler.y_data == []

        handler.load_data()

        self.assertEqual(expected_x_values, handler.x_data)
        self.assertEqual(expected_y_values, handler.y_data)

    def test_set_plotter(self):
        valid_num_epochs = 2
        handler = ConcreteSingleReplicateHandler("dummy_path", valid_num_epochs)

        plotter = handler.set_plotter()

        self.assertTrue(isinstance(plotter, MockPlotter))

    def test_generate_plot(self):
        valid_num_epochs = 2
        handler = ConcreteSingleReplicateHandler("dummy_path", valid_num_epochs)
        plotter = handler.set_plotter()

        handler.generate_plot(plotter)

        # the two tests below implicitly test that plotter.make_plot was called -- plotter.axis and plotter.figure are
        # None otherwise
        self.assertTrue(isinstance(plotter.axis, matplotlib.axes.SubplotBase))
        self.assertTrue(isinstance(plotter.figure, matplotlib.figure.Figure))

    def test_evaluate_experiment_show_and_save(self):
        if not show_no_plots_for_automated_tests:
            valid_num_epochs = 2
            path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
            assert os.path.exists(path_to_replicate)
            handler = ConcreteSingleReplicateHandler(path_to_replicate, valid_num_epochs)
            target_file = "./resources/test_plots/lottery_simplified/plots/ConcreteSingleReplicateHandler_plot.png"
            remove_target_file_if_exists(target_file)

            handler.evaluate_experiment(show_plot=True, save_plot=True)

            self.assertTrue(os.path.exists(target_file))
        else:
            print("Ignoring TestSparsityAccuracyOnSingleReplicateHandler.test_evaluate_experiment_show_and_save to "
                  "allow automated testing.")

    def test_evaluate_experiment_show_but_no_save(self):
        if not show_no_plots_for_automated_tests:
            valid_num_epochs = 2
            path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
            assert os.path.exists(path_to_replicate)
            handler = ConcreteSingleReplicateHandler(path_to_replicate, valid_num_epochs)
            target_file = "./resources/test_plots/lottery_simplified/plots/ConcreteSingleReplicateHandler_plot.png"
            remove_target_file_if_exists(target_file)

            handler.evaluate_experiment(show_plot=True, save_plot=False)

            self.assertFalse(os.path.exists(target_file))
        else:
            print("Ignoring TestSparsityAccuracyOnSingleReplicateHandler.test_evaluate_experiment_show_but_no_save to "
                  "allow automated testing.")

    def test_evaluate_experiment_no_show_but_save(self):
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert os.path.exists(path_to_replicate)
        handler = ConcreteSingleReplicateHandler(path_to_replicate, valid_num_epochs)
        target_file = "./resources/test_plots/lottery_simplified/plots/ConcreteSingleReplicateHandler_plot.png"
        remove_target_file_if_exists(target_file)

        handler.evaluate_experiment(show_plot=False, save_plot=True)

        self.assertTrue(os.path.exists(target_file))

    def test_evaluate_experiment_no_show_no_save(self):
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert os.path.exists(path_to_replicate)
        handler = ConcreteSingleReplicateHandler(path_to_replicate, valid_num_epochs)
        target_file = "./resources/test_plots/lottery_simplified/plots/ConcreteSingleReplicateHandler_plot.png"
        remove_target_file_if_exists(target_file)

        handler.evaluate_experiment(show_plot=False, save_plot=False)

        self.assertFalse(os.path.exists(target_file))


class ConcreteSingleReplicateHandler(experiment.SingleReplicateHandler):
    # mock implementation for testing non-abstract methods in SingleReplicateHandler.
    def load_x_data(self, paths):
        self.x_data = [1., 2., 3.]

    def load_y_data(self, paths):
        self.y_data = [1., 2., 3.]

    def prepare_data_for_plotting(self):
        pass

    def set_plotter(self):
        return MockPlotter()


class MockPlotter(plotters.PlotterBaseClass):
    title = "ConcreteSingleReplicateHandler title"
    x_label = "ConcreteSingleReplicateHandler x-label"
    y_label = "ConcreteSingleReplicateHandler y-label"
    save_file_name = "ConcreteSingleReplicateHandler_plot.png"

    def plot_data(self, axis, x_values, y_values):
        axis.plot(x_values, y_values)


def check_simplified_lottery_experiment_replicate_exists(path_to_replicate):
    if path_to_replicate[-1] != "/":
        path_to_replicate = path_to_replicate + "/"

    return (os.path.exists(path_to_replicate + "level_0/main/model_ep0_it0.pth") and
            os.path.exists(path_to_replicate + "level_0/main/model_ep2_it0.pth") and
            os.path.exists(path_to_replicate + "level_1/main/model_ep0_it0.pth") and
            os.path.exists(path_to_replicate + "level_1/main/model_ep2_it0.pth") and
            os.path.exists(path_to_replicate + "level_0/main/mask.pth") and
            os.path.exists(path_to_replicate + "level_1/main/mask.pth") and
            os.path.exists(path_to_replicate + "level_0/main/logger") and
            os.path.exists(path_to_replicate + "level_1/main/logger") and
            os.path.exists(path_to_replicate + "level_0/main/sparsity_report.json") and
            os.path.exists(path_to_replicate + "level_1/main/sparsity_report.json"))


def remove_target_file_if_exists(target_file):
    if os.path.exists(target_file):
        os.remove(target_file)
    assert not os.path.exists(target_file)


class TestSparsityAccuracyOnSingleReplicateHandler(unittest.TestCase):
    def test_is_subclass_of_SingleReplicateHandler(self):
        sparsity_accuracy_single_replicate_handler = experiment.SparsityAccuracyOnSingleReplicateHandler("dummy_path",
                                                                                                         1)
        self.assertTrue(issubclass(experiment.SparsityAccuracyOnSingleReplicateHandler,
                                   experiment.SingleReplicateHandler))
        self.assertTrue(isinstance(sparsity_accuracy_single_replicate_handler, experiment.SingleReplicateHandler))

    def test_load_x_data(self):  # loads sparsities
        expected_sparsities = [1.0, 212959.0/266200.0]
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        sparsity_accuracy_single_replicate_handler = experiment.SparsityAccuracyOnSingleReplicateHandler(
            path_to_replicate, valid_num_epochs)
        paths = sparsity_accuracy_single_replicate_handler.get_paths()

        sparsity_accuracy_single_replicate_handler.load_x_data(paths)

        self.assertEqual(expected_sparsities, sparsity_accuracy_single_replicate_handler.x_data)

    def test_load_y_data(self):  # loads accuracies
        expected_accuracies = [0.9644, 0.9678]
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        sparsity_accuracy_single_replicate_handler = experiment.SparsityAccuracyOnSingleReplicateHandler(
            path_to_replicate, valid_num_epochs)
        paths = sparsity_accuracy_single_replicate_handler.get_paths()

        sparsity_accuracy_single_replicate_handler.load_y_data(paths)

        self.assertEqual(expected_accuracies, sparsity_accuracy_single_replicate_handler.y_data)

    def test_prepare_data_for_plotting(self):
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        sparsity_accuracy_single_replicate_handler = \
            experiment.SparsityAccuracyOnSingleReplicateHandler(path_to_replicate, valid_num_epochs)
        sparsity_accuracy_single_replicate_handler.load_data()
        expected_sparsity = sparsity_accuracy_single_replicate_handler.x_data
        expected_accuracy = sparsity_accuracy_single_replicate_handler.y_data

        sparsity_accuracy_single_replicate_handler.prepare_data_for_plotting()

        self.assertEqual(expected_sparsity, sparsity_accuracy_single_replicate_handler.x_data)
        self.assertEqual(expected_accuracy, sparsity_accuracy_single_replicate_handler.y_data)

    def test_set_plotter(self):
        valid_num_epochs = 2
        sparsity_accuracy_single_replicate_handler = experiment.SparsityAccuracyOnSingleReplicateHandler(
            "dummy_path", valid_num_epochs)

        plotter = sparsity_accuracy_single_replicate_handler.set_plotter()

        self.assertTrue(isinstance(plotter, plotters.SparsityAccuracyReplicatePlotter))

    def test_evaluate_experiment(self):
        # sanity check
        if not show_no_plots_for_automated_tests:
            valid_num_epochs = 2
            path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
            assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
            sparsity_accuracy_single_replicate_handler = experiment.SparsityAccuracyOnSingleReplicateHandler(
                path_to_replicate, valid_num_epochs)

            sparsity_accuracy_single_replicate_handler.evaluate_experiment(True, False)


class TestSparsityNeuralPersistenceOnSingleReplicateHandler(unittest.TestCase):
    def test_is_subclass_of_SingleReplicateHandler(self):
        valid_num_epochs = 1
        sparsity_neural_persistence_single_replicate_handler = \
            experiment.SparsityNeuralPersistenceOnSingleReplicateHandler("dummy_path", valid_num_epochs)

        self.assertTrue(issubclass(experiment.SparsityNeuralPersistenceOnSingleReplicateHandler,
                                   experiment.SingleReplicateHandler))
        self.assertTrue(isinstance(sparsity_neural_persistence_single_replicate_handler,
                                   experiment.SingleReplicateHandler))

    def test_load_x_data(self):  # loads sparsities
        expected_sparsities = [1.0, 212959.0/266200.0]
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        sparsity_np_single_replicate_handler = experiment.SparsityNeuralPersistenceOnSingleReplicateHandler(
            path_to_replicate, valid_num_epochs)
        paths = sparsity_np_single_replicate_handler.get_paths()

        sparsity_np_single_replicate_handler.load_x_data(paths)

        self.assertEqual(expected_sparsities, sparsity_np_single_replicate_handler.x_data)

    def test_load_y_data(self):  # loads neural persistences
        expected_np_level_0, expected_np_level_1 = get_neural_persistences_for_lottery_simplified()
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        sparsity_np_single_replicate_handler = experiment.SparsityNeuralPersistenceOnSingleReplicateHandler(
            path_to_replicate, valid_num_epochs)
        paths = sparsity_np_single_replicate_handler.get_paths()

        sparsity_np_single_replicate_handler.load_y_data(paths)

        self.assertDictEqual(expected_np_level_0, sparsity_np_single_replicate_handler.y_data[0])
        self.assertDictEqual(expected_np_level_1, sparsity_np_single_replicate_handler.y_data[1])

    def test_prepare_data_for_plotting(self):
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
        sparsity_np_single_replicate_handler = experiment.SparsityNeuralPersistenceOnSingleReplicateHandler(
            path_to_replicate, valid_num_epochs)
        sparsity_np_single_replicate_handler.load_data()
        expected_sparsity = sparsity_np_single_replicate_handler.x_data
        expected_neural_persistence = utils.prepare_neural_persistence_for_plotting(
            sparsity_np_single_replicate_handler.y_data)

        sparsity_np_single_replicate_handler.prepare_data_for_plotting()

        self.assertEqual(expected_sparsity, sparsity_np_single_replicate_handler.x_data)
        self.assertEqual(expected_neural_persistence, sparsity_np_single_replicate_handler.y_data)

    def test_set_plotter(self):
        valid_num_epochs = 2
        sparsity_np_single_replicate_handler = experiment.SparsityNeuralPersistenceOnSingleReplicateHandler(
                "dummy_path", valid_num_epochs)

        plotter = sparsity_np_single_replicate_handler.set_plotter()

        self.assertTrue(isinstance(plotter, plotters.SparsityNeuralPersistenceReplicatePlotter))

    def test_evaluate_experiment(self):
        # sanity check
        if not show_no_plots_for_automated_tests:
            valid_num_epochs = 2
            path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
            assert check_simplified_lottery_experiment_replicate_exists(path_to_replicate)
            sparsity_np_single_replicate_handler = experiment.SparsityNeuralPersistenceOnSingleReplicateHandler(
                    path_to_replicate, valid_num_epochs)

            sparsity_np_single_replicate_handler.evaluate_experiment(True, False)


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


class TestSparsityAccuracyExperiment(unittest.TestCase):
    def test_sparsity_acc(self):
        expected_mean_accuracies = [np.mean([0.9644, 0.9544]), np.mean([0.9678, 0.9878])]
        expected_std_accuracies = [np.std([0.9644, 0.9544]), np.std([0.9678, 0.9878])]
        expected_sparsities = [1.0, 212959.0/266200.0]

        sparsities, mean_accuracies, std_accuracies = experiment.sparsity_accuracy_plot_experiment(
            "./resources/test_plots/lottery_simplified_experiment/", 2, show_plot=show_plot_off_for_fast_tests, save_plot=False)

        self.assertEqual(expected_sparsities, sparsities)
        self.assertEqual(expected_mean_accuracies, mean_accuracies)
        self.assertEqual(expected_std_accuracies, std_accuracies)

    def test_equal_sparsity_lengths(self):
        with self.assertRaises(AssertionError):
            experiment.sparsity_accuracy_plot_experiment(
                "./resources/test_plots/lottery_simplified_experiment_unequal_sparsities", 2,
                show_plot=show_plot_off_for_fast_tests, save_plot=False)

    def test_save_plot(self):
        if os.path.isfile("./resources/test_plots/lottery_simplified_experiment/plots/sparsity_accuracy_experiment.png"):
            os.remove("./resources/test_plots/lottery_simplified_experiment/plots/sparsity_accuracy_experiment.png")

        assert not os.path.exists(
            "./resources/test_plots/lottery_simplified_experiment/plots/sparsity_accuracy_experiment.png")

        _, _, _ = experiment.sparsity_accuracy_plot_experiment("./resources/test_plots/lottery_simplified_experiment/",
                                                               2, show_plot=show_plot_off_for_fast_tests, save_plot=True)
        self.assertTrue(os.path.isfile(
            "./resources/test_plots/lottery_simplified_experiment/plots/sparsity_accuracy_experiment.png"))


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
                "./resources/test_plots/lottery_simplified_experiment_unequal_sparsities", 2,
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
