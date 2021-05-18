import unittest

import numpy as np
import numpy.testing
import torch
import torch.nn as nn

import src.utils as utils


# simplified openLTH model
class TestModel(nn.Module):
    def __init__(self, plan, outputs=10):
        super(TestModel, self).__init__()

        layers = []
        current_size = 4
        for size in plan:
            layers.append(nn.Linear(current_size, size))
            current_size = size

        self.fc_layers = nn.ModuleList(layers)
        self.fc = nn.Linear(current_size, outputs)
        for layer in self.fc_layers:
            nn.init.constant_(layer.weight, -1)
        nn.init.constant_(self.fc.weight, -1)


class TestPrepareOrderedDictFromModel(unittest.TestCase):
    def test_on_vanilla_ff(self):

        model = TestModel([2], outputs=2)
        expected_keys = ['fc_layers.0.weight', 'fc.weight']
        expected_fc_layers_0_weight = torch.tensor([[-1., -1., -1., -1.], [-1., -1., -1., -1.]])
        expected_fc_weight = torch.tensor([[-1., -1.], [-1., -1.]])

        weights = utils.prepare_ordered_dict_from_model(model)

        self.assertEqual(expected_keys, list(weights.keys()))
        np.testing.assert_array_equal(expected_fc_layers_0_weight, weights['fc_layers.0.weight'])
        np.testing.assert_array_equal(expected_fc_weight, weights['fc.weight'])


class TestLoadMaskedWeights(unittest.TestCase):
    def test_loading_on_vanilla_ff(self):
        model = TestModel([2], outputs=2)
        torch.save(model, "./resources/test_load_masked_network/model_ep2_it0.pth")
        mask = {'fc_layers.0.weight': torch.tensor([[1., 0., 1., 0.], [0., 1., 1., 0.]]),
                'fc.weight': torch.tensor([[0., 0.], [1., 1.]])}
        torch.save({k: v.int() for k, v in mask.items()}, "./resources/test_load_masked_network/mask.pth")

        expected_keys = ['fc_layers.0.weight', 'fc.weight']
        expected_fc_layers_0_masked_weight = torch.tensor([[-1., 0., -1., 0.], [0., -1., -1., 0.]])
        expected_fc_masked_weight = torch.tensor([[0., 0.], [-1., -1.]])

        weights = utils.load_masked_weights("./resources/test_load_masked_network", 2)

        self.assertEqual(expected_keys, list(weights.keys()))
        numpy.testing.assert_array_equal(expected_fc_layers_0_masked_weight, weights["fc_layers.0.weight"])
        numpy.testing.assert_array_equal(expected_fc_masked_weight, weights["fc.weight"])


class TestGetFilePaths(unittest.TestCase):
    def test_train(self):
        train_path = "resources/test_get_file_paths/train_574e51abc295d8da78175b320504f2ba/"
        expected_results = [train_path + "replicate_1/main/" + "logger",
                            train_path + "replicate_1/main/" + "model_ep0_it0.pth",
                            train_path + "replicate_1/main/" + "model_ep40_it0.pth"]

        self.assertEqual(expected_results, utils.get_file_paths(train_path, "train", 40))
        self.assertEqual(expected_results, utils.get_file_paths(train_path[:-1], "train", 40))

    def test_lottery(self):
        lottery_path = "resources/test_get_file_paths/lottery_1db02943c54add91e13635735031a85e/"
        paths = [lottery_path + "replicate_1/level_{}/main/" + "logger",
                 lottery_path + "replicate_1/level_{}/main/" + "model_ep0_it0.pth",
                 lottery_path + "replicate_1/level_{}/main/" + "model_ep2_it0.pth"]
        expected_results = []
        for i in range(4):
            expected_results += list(map(lambda s: s.format(str(i)), paths))

        self.assertEqual(expected_results, utils.get_file_paths(lottery_path, "lottery", 2))

    def test_lottery_branch(self):
        valid_path = "resources/test_get_file_paths/temporary_lottery_branch_test/"
        valid_eps = 40
        # todo adapt experiment types test once this gets implemented.

        with(self.assertRaises(NotImplementedError)):
            utils.get_file_paths(valid_path, "lottery_branch", valid_eps)

    def test_invalid_inputs(self):
        valid_train_path = "resources/test_get_file_paths/train_574e51abc295d8da78175b320504f2ba/"
        valid_ex_type = "train"
        valid_eps = 40

        # test path
        with(self.assertRaises(FileNotFoundError)):
            utils.get_file_paths("invalid_path", valid_ex_type, valid_eps)
        with(self.assertRaises(FileNotFoundError)):
            invalid_existing_path = "resources/test_get_file_paths/train_123"
            utils.get_file_paths(invalid_existing_path, valid_ex_type, valid_eps)
        with(self.assertRaises(FileNotFoundError)):
            missing_logger = "./resources/test_get_file_paths/train_wo_logger"
            utils.get_file_paths(missing_logger, valid_ex_type, valid_eps)
        with(self.assertRaises(FileNotFoundError)):
            missing_model_ep0 = "./resources/test_get_file_paths/train_wo_model_ep0"
            utils.get_file_paths(missing_model_ep0, valid_ex_type, valid_eps)
        with(self.assertRaises(NotADirectoryError)):
            utils.get_file_paths(valid_train_path + "replicate_1/main/model_ep0_it0.pth", valid_ex_type,
                                 valid_eps)

        # test types
        with(self.assertRaises(ValueError)):
            utils.get_file_paths(valid_train_path, "invalid_type", valid_eps)
        with(self.assertRaises(ValueError)):
            utils.get_file_paths(valid_train_path, "lottery", valid_eps)
        # todo replace path with actual lottery_branch folder once I get to that
        with(self.assertRaises(ValueError)):
            utils.get_file_paths("resources/test_get_file_paths/temporary_lottery_branch_test",
                                 "lottery", valid_eps)

        # test eps
        with(self.assertRaises(FileNotFoundError)):
            utils.get_file_paths(valid_train_path, valid_ex_type, 4)


class TestLoadUnmaskedWeights(unittest.TestCase):
    def test_loading_on_vanilla_ff(self):
        model = TestModel([2], outputs=2)
        torch.save(model, "./resources/test_load_unmasked_network/model_ep0_it0.pth")
        expected_keys = ['fc_layers.0.weight', 'fc.weight']
        expected_fc_layers_0_weights = torch.tensor([[-1., -1., -1., -1.], [-1., -1., -1., -1.]])
        expected_fc_weights = torch.tensor([[-1., -1.], [-1., -1.]])

        weights = utils.load_unmasked_weights("./resources/test_load_unmasked_network/model_ep0_it0.pth")

        self.assertEqual(expected_keys, list(weights.keys()))
        np.testing.assert_array_equal(expected_fc_layers_0_weights, weights['fc_layers.0.weight'])
        np.testing.assert_array_equal(expected_fc_weights, weights['fc.weight'])


class TestLoadAccuracy(unittest.TestCase):
    def test_load_accuracy(self):
        expected_accuracy = 0.9644

        self.assertEqual(expected_accuracy, utils.load_accuracy("./resources/test_load_accuracy/logger"))

    def test_wanky_logger(self):
        with self.assertRaises(AssertionError):
            utils.load_accuracy("./resources/test_load_accuracy/logger_wrong_format")
        with self.assertRaises(AssertionError):
            utils.load_accuracy("./resources/test_load_accuracy/logger_too_large_accuracy")
        with self.assertRaises(AssertionError):
            utils.load_accuracy("./resources/test_load_accuracy/logger_negative_accuracy")


if __name__ == '__main__':
    unittest.main()
