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
from utils_tests import TestModel, generate_expected_paths_for_lottery_single_replicate_with_2_eps


show_plot_off_for_fast_tests = False

show_no_plots_for_automated_tests = True


class TestSparsityAccuracyOnSingleReplicateHandler(unittest.TestCase):
    def test_initialization(self):
        expected_path = "./dummy_test_path/"
        expected_epochs = 42
        expected_sparsities = []
        expected_accuracies = []

        handler = experiment.SparsityAccuracyOnSingleReplicateHandler(expected_path, expected_epochs)

        self.assertEqual(expected_path, handler.experiment_root_path)
        self.assertEqual(expected_epochs, handler.epochs)
        self.assertEqual(expected_sparsities, handler.sparsities)
        self.assertEqual(expected_accuracies, handler.accuracies)
        self.assertIsNone(handler.plotter)

    def test_initialization_with_invalid_epochs_raises(self):
        epoch_equals_0 = 0
        epoch_smaller_0 = -10
        epoch_no_integer = 3.8

        with self.assertRaises(ValueError):
            experiment.SparsityAccuracyOnSingleReplicateHandler("dummy_path", epoch_equals_0)
        with self.assertRaises(ValueError):
            experiment.SparsityAccuracyOnSingleReplicateHandler("dummy_path", epoch_smaller_0)
        with self.assertRaises(ValueError):
            experiment.SparsityAccuracyOnSingleReplicateHandler("dummy_path", epoch_no_integer)

    def test_raise_if_no_valid_epoch(self):
        valid_num_epochs = 1
        epoch_equals_0 = 0
        epoch_smaller_0 = -4
        epoch_no_integer = 4.5

        handler = experiment.SparsityAccuracyOnSingleReplicateHandler("dummy_path", valid_num_epochs)

        with self.assertRaises(ValueError):
            handler.raise_if_no_valid_epoch(epoch_equals_0)
        with self.assertRaises(ValueError):
            handler.raise_if_no_valid_epoch(epoch_smaller_0)
        with self.assertRaises(ValueError):
            handler.raise_if_no_valid_epoch(epoch_no_integer)

    def test_get_paths(self):
        lottery_path = "resources/test_get_paths_from_replicate/lottery_1db02943c54add91e13635735031a85e/replicate_1/"
        expected_result = generate_expected_paths_for_lottery_single_replicate_with_2_eps(lottery_path)
        num_epochs = 2
        sparsity_accuracy_single_replicate_handler = \
            experiment.SparsityAccuracyOnSingleReplicateHandler(lottery_path, num_epochs)

        actual_results = sparsity_accuracy_single_replicate_handler.get_paths()

        self.assertDictEqual(expected_result, actual_results)

    def test_load_sparsities(self):
        expected_sparsities = [1.0, 212959.0/266200.0]
        num_epochs = 2
        sparsity_accuracy_single_replicate_handler = \
            experiment.SparsityAccuracyOnSingleReplicateHandler("./resources/test_plots/lottery_simplified/replicate_1",
                                                                num_epochs)
        paths = sparsity_accuracy_single_replicate_handler.get_paths()

        sparsity_accuracy_single_replicate_handler.load_sparsities(paths)

        self.assertEqual(expected_sparsities, sparsity_accuracy_single_replicate_handler.sparsities)

    def test_load_accuracies(self):
        expected_accuracies = [0.9644, 0.9678]
        num_epochs = 2
        sparsity_accuracy_single_replicate_handler = \
            experiment.SparsityAccuracyOnSingleReplicateHandler("./resources/test_plots/lottery_simplified/replicate_1",
                                                                num_epochs)
        paths = sparsity_accuracy_single_replicate_handler.get_paths()

        sparsity_accuracy_single_replicate_handler.load_accuracies(paths)

        self.assertEqual(expected_accuracies, sparsity_accuracy_single_replicate_handler.accuracies)

    def test_prepare_data_for_plotting(self):
        expected_sparsities = [1.0, 212959.0/266200.0]
        expected_accuracies = [0.9644, 0.9678]
        num_epochs = 2
        sparsity_accuracy_single_replicate_handler = \
            experiment.SparsityAccuracyOnSingleReplicateHandler("./resources/test_plots/lottery_simplified/replicate_1",
                                                                num_epochs)
        assert sparsity_accuracy_single_replicate_handler.sparsities == [] and \
               sparsity_accuracy_single_replicate_handler.accuracies == []

        sparsity_accuracy_single_replicate_handler.prepare_data_for_plotting()

        self.assertEqual(expected_sparsities, sparsity_accuracy_single_replicate_handler.sparsities)
        self.assertEqual(expected_accuracies, sparsity_accuracy_single_replicate_handler.accuracies)

    def test_generate_plot(self):
        num_epochs = 2
        sparsity_accuracy_single_replicate_handler = \
            experiment.SparsityAccuracyOnSingleReplicateHandler("./resources/test_plots/lottery_simplified/replicate_1",
                                                                num_epochs)

        sparsity_accuracy_single_replicate_handler.generate_plot()

        self.assertTrue(isinstance(sparsity_accuracy_single_replicate_handler.plotter,
                                   plotters.SparsityAccuracyReplicatePlotter))
        # the two tests below implicitly test that plotter.make_plot was called -- plotter.axis and plotter.figure are
        # None otherwise
        self.assertTrue(isinstance(sparsity_accuracy_single_replicate_handler.plotter.axis,
                                   matplotlib.axes.SubplotBase))
        self.assertTrue(isinstance(sparsity_accuracy_single_replicate_handler.plotter.figure, matplotlib.figure.Figure))

    def test_evaluate_experiment_show_and_save(self):
        if not show_no_plots_for_automated_tests:
            num_epochs = 2
            handler = experiment.SparsityAccuracyOnSingleReplicateHandler(
                "./resources/test_plots/lottery_simplified/replicate_1", num_epochs)
            target_file = \
                "./resources/test_plots/lottery_simplified/plots/sparsity_accuracy_replicate_plot.png"
            remove_target_file_if_exists(target_file)
            assert not os.path.exists(target_file)

            handler.evaluate_experiment(show_plot=True, save_plot=True)

            self.assertTrue(os.path.exists(target_file))
            self.assertTrue(isinstance(handler.plotter.axis, matplotlib.axes.SubplotBase))
            self.assertTrue(isinstance(handler.plotter.figure, matplotlib.figure.Figure))
        else:
            print("Ignoring TestSparsityAccuracyOnSingleReplicateHandler.test_evaluate_experiment_show_and_save to "
                  "allow automated testing.")

    def test_evaluate_experiment_show_but_no_save(self):
        if not show_no_plots_for_automated_tests:
            num_epochs = 2
            handler = experiment.SparsityAccuracyOnSingleReplicateHandler(
                "./resources/test_plots/lottery_simplified/replicate_1", num_epochs)
            target_file = \
                "./resources/test_plots/lottery_simplified/plots/sparsity_accuracy_replicate_plot.png"
            remove_target_file_if_exists(target_file)
            assert not os.path.exists(target_file)

            handler.evaluate_experiment(show_plot=True, save_plot=False)

            self.assertFalse(os.path.exists(target_file))
            self.assertTrue(isinstance(handler.plotter.axis, matplotlib.axes.SubplotBase))
            self.assertTrue(isinstance(handler.plotter.figure, matplotlib.figure.Figure))
        else:
            print("Ignoring TestSparsityAccuracyOnSingleReplicateHandler.test_evaluate_experiment_show_but_no_save to "
                  "allow automated testing.")

    def test_evaluate_experiment_no_show_but_save(self):
        num_epochs = 2
        handler = experiment.SparsityAccuracyOnSingleReplicateHandler(
            "./resources/test_plots/lottery_simplified/replicate_1", num_epochs)
        target_file = \
            "./resources/test_plots/lottery_simplified/plots/sparsity_accuracy_replicate_plot.png"
        remove_target_file_if_exists(target_file)
        assert not os.path.exists(target_file)

        handler.evaluate_experiment(show_plot=False, save_plot=True)

        self.assertTrue(os.path.exists(target_file))

    def test_evaluate_experiment_no_show_no_save(self):
        num_epochs = 2
        handler = experiment.SparsityAccuracyOnSingleReplicateHandler(
            "./resources/test_plots/lottery_simplified/replicate_1", num_epochs)
        target_file = \
            "./resources/test_plots/lottery_simplified/plots/sparsity_accuracy_replicate_plot.png"
        remove_target_file_if_exists(target_file)
        assert not os.path.exists(target_file)

        handler.evaluate_experiment(show_plot=False, save_plot=False)

        self.assertFalse(os.path.exists(target_file))


def remove_target_file_if_exists(target_file):
    if os.path.exists(target_file):
        os.remove(target_file)


class TestSparsityAccuracyReplicate(unittest.TestCase):
    def test_sparsity_acc(self):
        expected_accuracies = [0.9644, 0.9678]
        expected_sparsities = [1.0, 212959.0/266200.0]

        sparsities, accuracies = experiment.sparsity_accuracy_plot_replicate("./resources/test_plots/lottery_simplified/replicate_1", 2,
                                                                             show_plot=show_plot_off_for_fast_tests, save_plot=False)
        self.assertEqual(expected_sparsities, sparsities)
        self.assertEqual(expected_accuracies, accuracies)

    def test_save_plot(self):
        if os.path.isfile("./resources/test_plots/lottery_simplified/plots/sparsity_accuracy.png"):
            os.remove("./resources/test_plots/lottery_simplified/plots/sparsity_accuracy.png")

        assert not os.path.exists("./resources/test_plots/lottery_simplified/plots/sparsity_accuracy.png")

        _, _ = experiment.sparsity_accuracy_plot_replicate("./resources/test_plots/lottery_simplified/replicate_1", 2,
                                                           show_plot=show_plot_off_for_fast_tests, save_plot=True)
        self.assertTrue(os.path.isfile("./resources/test_plots/lottery_simplified/plots/sparsity_accuracy.png"))


class TestSparsityNeuralPersistenceReplicate(unittest.TestCase):
    def test_sparsity_np(self):
        setup_simplified_lottery_experiment()

        expected_dict_level_0 = OrderedDict([('fc_layers.0.weight', torch.tensor([[-1., -1., -1., -1.],
                                                                                  [-1., -1., -1., -1.]]).numpy()),
                                             ('fc.weight', torch.tensor([[-1., -1.], [-1., -1.]]).numpy())])
        expected_dict_level_1 = OrderedDict([('fc_layers.0.weight', torch.tensor([[-1., 0., -1., 0.],
                                                                                  [0., -1., -1., 0.]]).numpy()),
                                             ('fc.weight', torch.tensor([[0., 0.], [-1., -1.]]).numpy())])
        per_layer_calc = PerLayerCalculation()

        expected_sparsities = [1.0, 212959.0/266200.0]

        sparsities, neural_pers = experiment.sparsity_neural_persistence_plot_replicate(
            "./resources/test_plots/lottery_simplified/replicate_1", 2, show_plot=show_plot_off_for_fast_tests, save_plot=False)

        self.assertEqual(expected_sparsities, sparsities)
        self.assertDictEqual(per_layer_calc(expected_dict_level_0), neural_pers[0])
        self.assertDictEqual(per_layer_calc(expected_dict_level_1), neural_pers[1])

    def test_save_plot(self):
        if os.path.isfile("./resources/test_plots/lottery_simplified/plots/sparsity_neural_persistence.png"):
            os.remove("./resources/test_plots/lottery_simplified/plots/sparsity_neural_persistence.png")

        assert not os.path.exists("./resources/test_plots/lottery_simplified/plots/sparsity_neural_persistence.png")

        _, _ = experiment.sparsity_neural_persistence_plot_replicate("./resources/test_plots/lottery_simplified/replicate_1", 2,
                                                                     show_plot=show_plot_off_for_fast_tests, save_plot=True)
        self.assertTrue(os.path.isfile(
            "./resources/test_plots/lottery_simplified/plots/sparsity_neural_persistence.png"))


def setup_simplified_lottery_experiment():
    model = TestModel([2], outputs=2)
    torch.save(OrderedDict(model.named_parameters()),
               "./resources/test_plots/lottery_simplified/replicate_1/level_0/main/model_ep0_it0.pth")
    torch.save(OrderedDict(model.named_parameters()),
               "./resources/test_plots/lottery_simplified/replicate_1/level_0/main/model_ep2_it0.pth")
    torch.save(OrderedDict(model.named_parameters()),
               "./resources/test_plots/lottery_simplified/replicate_1/level_1/main/model_ep0_it0.pth")
    torch.save(OrderedDict(model.named_parameters()),
               "./resources/test_plots/lottery_simplified/replicate_1/level_1/main/model_ep2_it0.pth")

    mask_level_0 = {'fc_layers.0.weight': torch.tensor([[1., 1., 1., 1.], [1., 1., 1., 1.]]),
                    'fc.weight': torch.tensor([[1., 1.], [1., 1.]])}
    mask_level_1 = {'fc_layers.0.weight': torch.tensor([[1., 0., 1., 0.], [0., 1., 1., 0.]]),
                    'fc.weight': torch.tensor([[0., 0.], [1., 1.]])}
    torch.save({k: v.int() for k, v in mask_level_0.items()},
               "./resources/test_plots/lottery_simplified/replicate_1/level_0/main/mask.pth")
    torch.save({k: v.int() for k, v in mask_level_1.items()},
               "./resources/test_plots/lottery_simplified/replicate_1/level_1/main/mask.pth")


class TestAccuracyNeuralPersistenceReplicate(unittest.TestCase):
    def test_acc_np(self):
        correct_eps = 2
        expected_accuracies = [0.9644, 0.9678]

        setup_simplified_lottery_experiment()

        expected_dict_level_0 = OrderedDict([('fc_layers.0.weight', torch.tensor([[-1., -1., -1., -1.],
                                                                                  [-1., -1., -1., -1.]]).numpy()),
                                             ('fc.weight', torch.tensor([[-1., -1.], [-1., -1.]]).numpy())])
        expected_dict_level_1 = OrderedDict([('fc_layers.0.weight', torch.tensor([[-1., 0., -1., 0.],
                                                                                  [0., -1., -1., 0.]]).numpy()),
                                             ('fc.weight', torch.tensor([[0., 0.], [-1., -1.]]).numpy())])
        per_layer_calc = PerLayerCalculation()

        accuracies, neural_pers = experiment.accuracy_neural_persistence_plot_replicate(
            "./resources/test_plots/lottery_simplified/replicate_1", correct_eps, show_plot=show_plot_off_for_fast_tests, save_plot=False)

        self.assertEqual(expected_accuracies, accuracies)

        self.assertDictEqual(per_layer_calc(expected_dict_level_0), neural_pers[0])
        self.assertDictEqual(per_layer_calc(expected_dict_level_1), neural_pers[1])

    def test_save_plot(self):
        if os.path.isfile("./resources/test_plots/lottery_simplified/plots/accuracy_neural_persistence.png"):
            os.remove("./resources/test_plots/lottery_simplified/plots/accuracy_neural_persistence.png")

        assert not os.path.exists("./resources/test_plots/lottery_simplified/plots/accuracy_neural_persistence.png")

        _, _ = experiment.accuracy_neural_persistence_plot_replicate("./resources/test_plots/lottery_simplified/replicate_1", 2,
                                                                     show_plot=show_plot_off_for_fast_tests, save_plot=True)
        self.assertTrue(os.path.isfile(
            "./resources/test_plots/lottery_simplified/plots/accuracy_neural_persistence.png"))


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
