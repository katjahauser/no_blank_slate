from collections import OrderedDict
import unittest

import torch

import src.experiment as experiment
from utils_tests import TestModel
from deps.neural_persistence.src.tda import PerLayerCalculation


def setup_simplified_lottery_experiment():
    model = TestModel([2], outputs=2)
    torch.save(model, "./resources/test_plots/lottery_simplified/replicate_1/level_0/main/model_ep0_it0.pth")
    torch.save(model, "./resources/test_plots/lottery_simplified/replicate_1/level_0/main/model_ep2_it0.pth")
    torch.save(model, "./resources/test_plots/lottery_simplified/replicate_1/level_1/main/model_ep0_it0.pth")
    torch.save(model, "./resources/test_plots/lottery_simplified/replicate_1/level_1/main/model_ep2_it0.pth")

    mask_level_0 = {'fc_layers.0.weight': torch.tensor([[1., 1., 1., 1.], [1., 1., 1., 1.]]),
                    'fc.weight': torch.tensor([[1., 1.], [1., 1.]])}
    mask_level_1 = {'fc_layers.0.weight': torch.tensor([[1., 0., 1., 0.], [0., 1., 1., 0.]]),
                    'fc.weight': torch.tensor([[0., 0.], [1., 1.]])}
    torch.save({k: v.int() for k, v in mask_level_0.items()},
               "./resources/test_plots/lottery_simplified/replicate_1/level_0/mask.pth")
    torch.save({k: v.int() for k, v in mask_level_1.items()},
               "./resources/test_plots/lottery_simplified/replicate_1/level_1/mask.pth")


class TestSparsityAccuracy(unittest.TestCase):
    def test_sparsity_acc(self):
        correct_eps = 2
        accuracies = [0.9644, 0.9678]

        setup_simplified_lottery_experiment()

        expected_dict_level_0 = OrderedDict([('fc_layers.0.weight', torch.tensor([[-1., -1., -1., -1.],
                                                                                  [-1., -1., -1., -1.]]).numpy()),
                                             ('fc.weight', torch.tensor([[-1., -1.], [-1., -1.]]).numpy())])
        expected_dict_level_1 = OrderedDict([('fc_layers.0.weight', torch.tensor([[-1., 0., -1., 0.],
                                                                                  [0., -1., -1., 0.]]).numpy()),
                                             ('fc.weight', torch.tensor([[0., 0.], [-1., -1.]]).numpy())])
        per_layer_calc = PerLayerCalculation()

        self.assertDictEqual(per_layer_calc(expected_dict_level_0),
                             experiment.sparsity_accuracy("./resources/test_plots/lottery_simplified", correct_eps)[0])
        self.assertDictEqual(per_layer_calc(expected_dict_level_1),
                             experiment.sparsity_accuracy("./resources/test_plots/lottery_simplified", correct_eps)[1])


class TestNeuralPersistenceSparsity(unittest.TestCase):
    def test_np_sparsity(self):
        experiment.neural_persistence_sparsity("dummypath")


class TestAccuracyNeuralPersistence(unittest.TestCase):
    def test_acc_np(self):
        experiment.accuracy_neural_persistence("dummypath")


if __name__ == '__main__':
    unittest.main()
