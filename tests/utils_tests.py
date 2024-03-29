from collections import OrderedDict
import unittest

import numpy as np
import numpy.testing
import torch
import torch.nn as nn

import src.utils as utils
import experiments_tests


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

        weights = utils.prepare_ordered_dict_from_model(OrderedDict(model.named_parameters()))

        self.assertEqual(expected_keys, list(weights.keys()))
        np.testing.assert_array_equal(expected_fc_layers_0_weight, weights['fc_layers.0.weight'])
        np.testing.assert_array_equal(expected_fc_weight, weights['fc.weight'])


class TestLoadMaskedWeights(unittest.TestCase):
    def test_loading_on_vanilla_ff(self):
        model = TestModel([2], outputs=2)
        torch.save(OrderedDict(model.named_parameters()), "./resources/test_load_masked_network/model_ep2_it0.pth")
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


class TestGetPathsFromReplicate(unittest.TestCase):
    def test_train(self):
        train_path = "resources/test_get_paths_from_replicate/train_574e51abc295d8da78175b320504f2ba/replicate_1/"

        expected_result = {"accuracy": train_path + "main/" + "logger",
                           "model_start": train_path + "main/" + "model_ep0_it0.pth",
                           "model_end": train_path + "main/" + "model_ep40_it0.pth"}

        self.assertDictEqual(expected_result, utils.get_paths_from_replicate(train_path, "train", 40))
        self.assertDictEqual(expected_result, utils.get_paths_from_replicate(train_path[:-1], "train", 40))

    def test_lottery(self):
        lottery_path = "resources/test_get_paths_from_replicate/lottery_1db02943c54add91e13635735031a85e/replicate_1/"

        expected_result = generate_expected_paths_for_lottery_replicate(lottery_path, num_levels=4)

        actual_result = utils.get_paths_from_replicate(lottery_path, "lottery", 2)

        self.assertDictEqual(expected_result, actual_result)

    def test_lottery_branch(self):
        valid_path = "resources/test_get_paths_from_replicate/temporary_lottery_branch_test/replicate_1/"
        valid_eps = 40
        # todo adapt experiment types test once this gets implemented.

        with(self.assertRaises(NotImplementedError)):
            utils.get_paths_from_replicate(valid_path, "lottery_branch", valid_eps)

    def test_invalid_inputs(self):
        valid_train_path = "resources/test_get_paths_from_replicate/train_574e51abc295d8da78175b320504f2ba/replicate_1/"
        valid_ex_type = "train"
        valid_eps = 40

        # test path parameter
        with(self.assertRaises(FileNotFoundError)):
            utils.get_paths_from_replicate("invalid_path", valid_ex_type, valid_eps)
        with(self.assertRaises(FileNotFoundError)):
            invalid_existing_path = "resources/test_get_paths_from_replicate/train_123"
            utils.get_paths_from_replicate(invalid_existing_path, valid_ex_type, valid_eps)
        with(self.assertRaises(FileNotFoundError)):
            missing_logger = "./resources/test_get_paths_from_replicate/train_wo_logger"
            utils.get_paths_from_replicate(missing_logger, valid_ex_type, valid_eps)
        with(self.assertRaises(FileNotFoundError)):
            missing_report = "./resources/test_get_paths_from_replicate/lottery_wo_sparsity_report"
            utils.get_paths_from_replicate(missing_report, "lottery", 2)
        with(self.assertRaises(FileNotFoundError)):
            missing_model_ep0 = "./resources/test_get_paths_from_replicate/train_wo_model_ep0"
            utils.get_paths_from_replicate(missing_model_ep0, valid_ex_type, valid_eps)
        with(self.assertRaises(NotADirectoryError)):
            not_a_directory = valid_train_path + "main/model_ep0_it0.pth"
            utils.get_paths_from_replicate(not_a_directory, valid_ex_type, valid_eps)

        # test type parameter
        with(self.assertRaises(ValueError)):
            utils.get_paths_from_replicate(valid_train_path, "invalid_type", valid_eps)
        with(self.assertRaises(ValueError)):
            utils.get_paths_from_replicate(valid_train_path, "lottery", valid_eps)
        # todo replace path with actual lottery_branch folder once I get to that
        with(self.assertRaises(ValueError)):
            utils.get_paths_from_replicate("resources/test_get_paths_from_replicate/temporary_lottery_branch_test",
                                           "lottery", valid_eps)

        # test eps parameter
        with(self.assertRaises(FileNotFoundError)):
            utils.get_paths_from_replicate(valid_train_path, valid_ex_type, 4)


def generate_expected_paths_for_lottery_replicate(replicate_path, num_levels):
    if replicate_path[-1] != "/":
        replicate_path = replicate_path + "/"

    paths = [replicate_path + "level_{}/main/" + "logger",
             replicate_path + "level_{}/main/" + "sparsity_report.json",
             replicate_path + "level_{}/main/" + "model_ep0_it0.pth",
             replicate_path + "level_{}/main/" + "model_ep2_it0.pth",
             replicate_path + "level_{}/main/" + "mask.pth"]
    expected_result = {"accuracy": [], "sparsity": [], "model_start": [], "model_end": []}
    for level_num in range(num_levels):
        expected_result["accuracy"].append(paths[0].format(str(level_num)))
        expected_result["sparsity"].append(paths[1].format(str(level_num)))
        expected_result["model_start"].append(paths[2].format(str(level_num)))
        if level_num == 0:
            expected_result["model_end"].append((paths[3].format(str(level_num)), None))
        else:
            expected_result["model_end"].append((paths[3].format(str(level_num)), paths[4].format(str(level_num))))
    return expected_result


class TestGetPathsFromExperiment(unittest.TestCase):
    # todo add check that all experiments have the same number of pruning steps
    def test_loading_lottery_paths(self):
        # the directory behind lottery_path contains a placeholder for the plot directory to test that these paths are
        # not considered
        lottery_path = "resources/test_get_paths_from_experiment/lottery_1db02943c54add91e13635735031a85e/"

        expected_result = generate_expected_paths_for_lottery_experiment(lottery_path, num_replicates=2, num_levels=4)

        self.assertDictEqual(expected_result, utils.get_paths_from_experiment(lottery_path, "lottery", 2))
        # test correct behaviour for paths without trailing slash
        self.assertDictEqual(expected_result, utils.get_paths_from_experiment(lottery_path[:-1], "lottery", 2))

    def test_loading_train_paths(self):
        with self.assertRaises(NotImplementedError):
            utils.get_paths_from_experiment("resources/test_get_paths_from_experiment/train_placeholder", "train", -1)

    def test_loading_lottery_branch_paths(self):
        with self.assertRaises(NotImplementedError):
            utils.get_paths_from_experiment("resources/test_get_paths_from_experiment/lottery_branch_placeholder",
                                            "lottery_branch", -1)

    def test_raises_when_unequal_number_of_pruning_steps(self):
        with self.assertRaises(AssertionError):
            utils.get_paths_from_experiment(
                "./resources/test_get_paths_from_experiment/lottery_w_different_pruning_levels", "lottery", 2)

    def test_invalid_inputs(self):
        valid_path = "resources/test_get_paths_from_experiment/lottery_1db02943c54add91e13635735031a85e"
        valid_eps = 2
        with self.assertRaises(ValueError):
            utils.get_paths_from_experiment(valid_path, "invalid_experiment_type", valid_eps)

        with self.assertRaises(FileNotFoundError):
            utils.get_paths_from_experiment("invalid_path", "train", valid_eps)

        with self.assertRaises(FileNotFoundError):
            utils.get_paths_from_experiment("resources/test_get_paths_from_experiment/no_replicates", "lottery",
                                            valid_eps)


def generate_expected_paths_for_lottery_experiment(lottery_path, num_replicates, num_levels):
    if lottery_path[-1] != "/":
        lottery_path = lottery_path + "/"
    paths = [lottery_path + "replicate_{}/level_{}/main/" + "logger",
             lottery_path + "replicate_{}/level_{}/main/" + "sparsity_report.json",
             lottery_path + "replicate_{}/level_{}/main/" + "model_ep0_it0.pth",
             lottery_path + "replicate_{}/level_{}/main/" + "model_ep2_it0.pth",
             lottery_path + "replicate_{}/level_{}/main/" + "mask.pth"]
    expected_result = {"replicate_1": {"accuracy": [], "sparsity": [], "model_start": [], "model_end": []},
                       "replicate_2": {"accuracy": [], "sparsity": [], "model_start": [], "model_end": []}}
    for replicate_num in range(1, num_replicates + 1):
        for level_num in range(num_levels):
            expected_result["replicate_{}".format(str(replicate_num))]["accuracy"].append(
                paths[0].format(str(replicate_num), str(level_num)))
            expected_result["replicate_{}".format(str(replicate_num))]["sparsity"].append(
                paths[1].format(str(replicate_num), str(level_num)))
            expected_result["replicate_{}".format(str(replicate_num))]["model_start"].append(
                paths[2].format(str(replicate_num), str(level_num)))
            if level_num == 0:
                expected_result["replicate_{}".format(str(replicate_num))]["model_end"].append(
                    (paths[3].format(str(replicate_num), str(level_num)), None))
            else:
                expected_result["replicate_{}".format(str(replicate_num))]["model_end"].append(
                    (paths[3].format(str(replicate_num), str(level_num)), paths[4].format(str(replicate_num),
                                                                                          str(level_num))))
    return expected_result


class TestLoadUnmaskedWeights(unittest.TestCase):
    def test_loading_on_vanilla_ff(self):
        model = TestModel([2], outputs=2)
        torch.save(OrderedDict(model.named_parameters()), "./resources/test_load_unmasked_network/model_ep0_it0.pth")
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

    def test_faulty_sparsity(self):
        with self.assertRaises(AssertionError):
            utils.load_sparsity("./resources/test_load_sparsity/sparsity_report_more_unpruned_then_there.json")


class TestPrepareNeuralPersistencesForPlotting(unittest.TestCase):
    def test_prepare_np_for_plotting(self):
        neural_pers = [{'fc_layers.0.weight': {'total_persistence': 1.0,
                                               'total_persistence_normalized': 0.4472135954999579},
                        'fc.weight': {'total_persistence': 1.0,
                                      'total_persistence_normalized': 0.5773502691896258},
                        'global': {'accumulated_total_persistence': 2.0,
                                   'accumulated_total_persistence_normalized': 0.5122819323447919}},
                       {'fc_layers.0.weight': {'total_persistence': 1.4142135623730951,
                                               'total_persistence_normalized': 0.6324555320336759},
                        'fc.weight': {'total_persistence': 1.4142135623730951,
                                      'total_persistence_normalized': 0.8164965809277261},
                        'global': {'accumulated_total_persistence': 2.8284271247461903,
                                   'accumulated_total_persistence_normalized': 0.724476056480701}}]
        expected_np = {'fc_layers.0.weight': [0.4472135954999579, 0.6324555320336759],
                       'fc.weight': [0.5773502691896258, 0.8164965809277261],
                       'global': [0.5122819323447919, 0.724476056480701]}

        self.assertDictEqual(expected_np, utils.prepare_neural_persistence_for_plotting(neural_pers))


class TestLoadingFunctionsForReplicates(unittest.TestCase):
    def test_load_sparsities_of_replicate(self):
        expected_sparsities = [1.0, 212959.0/266200.0]
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        paths = utils.get_paths_from_replicate(path_to_replicate, "lottery", valid_num_epochs)

        actual_sparsities = utils.load_sparsities_of_replicate(paths)

        self.assertEqual(expected_sparsities, actual_sparsities)

    def test_load_accuracies_of_replicate(self):
        expected_accuracies = [0.9644, 0.9678]
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        paths = utils.get_paths_from_replicate(path_to_replicate, "lottery", valid_num_epochs)

        actual_accuracies = utils.load_accuracies_of_replicate(paths)

        self.assertEqual(expected_accuracies, actual_accuracies)

    def test_load_neural_persistences_of_replicate(self):
        expected_np_level_0, expected_np_level_1 = experiments_tests.get_neural_persistences_for_lottery_simplified()
        valid_num_epochs = 2
        path_to_replicate = "./resources/test_plots/lottery_simplified/replicate_1"
        paths = utils.get_paths_from_replicate(path_to_replicate, "lottery", valid_num_epochs)

        actual_nps = utils.load_neural_persistences_of_replicate(paths)

        self.assertDictEqual(expected_np_level_0, actual_nps[0])
        self.assertDictEqual(expected_np_level_1, actual_nps[1])


class TestLoadingFunctionsForExperiment(unittest.TestCase):
    def test_load_sparsities_of_experiment(self):
        expected_sparsities = [1.0, 212959.0/266200.0]
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        paths = utils.get_paths_from_experiment(experiment_path, "lottery", valid_epochs)

        actual_sparsities = utils.load_sparsities_of_experiment(paths)

        self.assertEqual(expected_sparsities, actual_sparsities)

    def test_load_sparsities_on_unequal_sparsities_raises(self):
        experiment_path = "./resources/test_get_paths_from_experiment/lottery_simplified_experiment_unequal_sparsities"
        valid_epochs = 2
        paths = utils.get_paths_from_experiment(experiment_path, "lottery", valid_epochs)

        with self.assertRaises(AssertionError):
            utils.load_sparsities_of_experiment(paths)

    def test_assert_sparsities_equal_raises_on_unequal_sparsities(self):
        sparsities1 = [1., .2]
        sparsities2 = [1., .1]

        with self.assertRaises(AssertionError):
            utils.assert_sparsities_equal(sparsities1, sparsities2, "key1", "key2")

    def test_assert_sparsities_equal_does_not_raise_on_equal_sparsities(self):
        sparsities1 = [1., .1]
        sparsities2 = [1., .1]

        should_be_none = utils.assert_sparsities_equal(sparsities1, sparsities2, "key1", "key2")

        self.assertIsNone(should_be_none)

    def test_load_accuracies_of_experiment(self):
        expected_accuracies = np.asarray([[0.9644, 0.9678], [0.9544, 0.9878]])
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        paths = utils.get_paths_from_experiment(experiment_path, "lottery", valid_epochs)

        actual_accuracies = utils.load_accuracies_of_experiment(paths)

        np.testing.assert_array_equal(expected_accuracies, actual_accuracies)

    def test_load_neural_persistences_of_experiment(self):
        expected_np_level_0, expected_np_level_1 = experiments_tests.get_neural_persistences_for_lottery_simplified()
        expected_nps = [[expected_np_level_0, expected_np_level_1], [expected_np_level_0, expected_np_level_1]]
        experiment_path = "./resources/test_plots/lottery_simplified_experiment"
        valid_epochs = 2
        paths = utils.get_paths_from_experiment(experiment_path, "lottery", valid_epochs)

        actual_nps = utils.load_neural_persistences_of_experiment(paths)

        self.assertEqual(expected_nps, actual_nps)
        for i in range(len(expected_nps)):
            self.assertDictEqual(expected_nps[i][0], actual_nps[i][0])
            self.assertDictEqual(expected_nps[i][1], actual_nps[i][1])


if __name__ == '__main__':
    unittest.main()
