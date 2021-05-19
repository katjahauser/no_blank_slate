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

        weights = utils.load_masked_weights("./resources/test_load_masked_network/model_ep2_it0.pth",
                                            "./resources/test_load_masked_network/mask.pth")

        self.assertEqual(expected_keys, list(weights.keys()))
        numpy.testing.assert_array_equal(expected_fc_layers_0_masked_weight, weights["fc_layers.0.weight"])
        numpy.testing.assert_array_equal(expected_fc_masked_weight, weights["fc.weight"])


class TestGetFilePaths(unittest.TestCase):
    def test_train(self):
        train_path = "resources/test_get_file_paths/train_574e51abc295d8da78175b320504f2ba/"

        expected_result = {"logger": train_path + "replicate_1/main/" + "logger",
                           "model_start": train_path + "replicate_1/main/" + "model_ep0_it0.pth",
                           "model_end": train_path + "replicate_1/main/" + "model_ep40_it0.pth"}

        self.assertDictEqual(expected_result, utils.get_file_paths(train_path, "train", 40))
        self.assertDictEqual(expected_result, utils.get_file_paths(train_path[:-1], "train", 40))

    def test_lottery(self):
        lottery_path = "resources/test_get_file_paths/lottery_1db02943c54add91e13635735031a85e/"
        paths = [lottery_path + "replicate_1/level_{}/main/" + "logger",
                 lottery_path + "replicate_1/level_{}/main/" + "sparsity_report.json",
                 lottery_path + "replicate_1/level_{}/main/" + "model_ep0_it0.pth",
                 lottery_path + "replicate_1/level_{}/main/" + "model_ep2_it0.pth",
                 lottery_path + "replicate_1/level_{}/main/" + "mask.pth"]
        expected_result = {"logger": [], "sparsity_report": [], "model_start": [], "model_end": []}
        for i in range(4):
            expected_result["logger"].append(paths[0].format(str(i)))
            expected_result["sparsity_report"].append(paths[1].format(str(i)))
            expected_result["model_start"].append(paths[2].format(str(i)))
            if i == 0:
                expected_result["model_end"].append((paths[3].format(str(i)), None))
            else:
                expected_result["model_end"].append((paths[3].format(str(i)), paths[4].format(str(i))))

        self.assertDictEqual(expected_result, utils.get_file_paths(lottery_path, "lottery", 2))

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

        # test path parameter
        with(self.assertRaises(FileNotFoundError)):
            utils.get_file_paths("invalid_path", valid_ex_type, valid_eps)
        with(self.assertRaises(FileNotFoundError)):
            invalid_existing_path = "resources/test_get_file_paths/train_123"
            utils.get_file_paths(invalid_existing_path, valid_ex_type, valid_eps)
        with(self.assertRaises(FileNotFoundError)):
            missing_logger = "./resources/test_get_file_paths/train_wo_logger"
            utils.get_file_paths(missing_logger, valid_ex_type, valid_eps)
        with(self.assertRaises(FileNotFoundError)):
            missing_report = "./resources/test_get_file_paths/lottery_wo_sparsity_report"
            utils.get_file_paths(missing_report, "lottery", 2)
        with(self.assertRaises(FileNotFoundError)):
            missing_model_ep0 = "./resources/test_get_file_paths/train_wo_model_ep0"
            utils.get_file_paths(missing_model_ep0, valid_ex_type, valid_eps)
        with(self.assertRaises(NotADirectoryError)):
            not_a_directory = valid_train_path + "replicate_1/main/model_ep0_it0.pth"
            utils.get_file_paths(not_a_directory, valid_ex_type, valid_eps)

        # test type parameter
        with(self.assertRaises(ValueError)):
            utils.get_file_paths(valid_train_path, "invalid_type", valid_eps)
        with(self.assertRaises(ValueError)):
            utils.get_file_paths(valid_train_path, "lottery", valid_eps)
        # todo replace path with actual lottery_branch folder once I get to that
        with(self.assertRaises(ValueError)):
            utils.get_file_paths("resources/test_get_file_paths/temporary_lottery_branch_test",
                                 "lottery", valid_eps)

        # test eps parameter
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


class TestLoadSparsity(unittest.TestCase):
    def test_load_sparsity(self):
        expected_sparsity = 170366.0/266200.0

        self.assertEqual(expected_sparsity, utils.load_sparsity("./resources/test_load_sparsity/sparsity_report.json"))

    def test_wanky_sparsity(self):
        with self.assertRaises(AssertionError):
            utils.load_sparsity("./resources/test_load_sparsity/sparsity_report_more_unpruned_then_there.json")


if __name__ == '__main__':
    unittest.main()
