from collections import OrderedDict
import os.path
import unittest

import torch

from deps.neural_persistence.src.tda import PerLayerCalculation
import src.experiment as experiment
from utils_tests import TestModel


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


class TestSparsityAccuracyReplicate(unittest.TestCase):
    def test_sparsity_acc(self):
        expected_accuracies = [0.9644, 0.9678]
        expected_sparsities = [1.0, 212959.0/266200.0]

        sparsities, accuracies = experiment.sparsity_accuracy_plot_replicate("./resources/test_plots/lottery_simplified/replicate_1", 2,
                                                                             show_plot=False, save_plot=False)
        self.assertEqual(expected_sparsities, sparsities)
        self.assertEqual(expected_accuracies, accuracies)

    def test_save_plot(self):
        if os.path.isfile("./resources/test_plots/lottery_simplified/plots/sparsity_accuracy.png"):
            os.remove("./resources/test_plots/lottery_simplified/plots/sparsity_accuracy.png")

        assert not os.path.exists("./resources/test_plots/lottery_simplified/plots/sparsity_accuracy.png")

        _, _ = experiment.sparsity_accuracy_plot_replicate("./resources/test_plots/lottery_simplified/replicate_1", 2,
                                                           show_plot=False, save_plot=True)
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
            "./resources/test_plots/lottery_simplified/replicate_1", 2)

        self.assertEqual(expected_sparsities, sparsities)
        self.assertDictEqual(per_layer_calc(expected_dict_level_0), neural_pers[0])
        self.assertDictEqual(per_layer_calc(expected_dict_level_1), neural_pers[1])

    def test_save_plot(self):
        if os.path.isfile("./resources/test_plots/lottery_simplified/plots/sparsity_neural_persistence.png"):
            os.remove("./resources/test_plots/lottery_simplified/plots/sparsity_neural_persistence.png")

        assert not os.path.exists("./resources/test_plots/lottery_simplified/plots/sparsity_neural_persistence.png")

        _, _ = experiment.sparsity_neural_persistence_plot_replicate("./resources/test_plots/lottery_simplified/replicate_1", 2,
                                                                     show_plot=False, save_plot=True)
        self.assertTrue(os.path.isfile(
            "./resources/test_plots/lottery_simplified/plots/sparsity_neural_persistence.png"))


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
            "./resources/test_plots/lottery_simplified/replicate_1", correct_eps, show_plot=False, save_plot=False)

        self.assertEqual(expected_accuracies, accuracies)

        self.assertDictEqual(per_layer_calc(expected_dict_level_0), neural_pers[0])
        self.assertDictEqual(per_layer_calc(expected_dict_level_1), neural_pers[1])

    def test_save_plot(self):
        if os.path.isfile("./resources/test_plots/lottery_simplified/plots/accuracy_neural_persistence.png"):
            os.remove("./resources/test_plots/lottery_simplified/plots/accuracy_neural_persistence.png")

        assert not os.path.exists("./resources/test_plots/lottery_simplified/plots/accuracy_neural_persistence.png")

        _, _ = experiment.accuracy_neural_persistence_plot_replicate("./resources/test_plots/lottery_simplified/replicate_1", 2,
                                                                     show_plot=False, save_plot=True)
        self.assertTrue(os.path.isfile(
            "./resources/test_plots/lottery_simplified/plots/accuracy_neural_persistence.png"))


if __name__ == '__main__':
    unittest.main()
