from collections import OrderedDict
import os.path

import json
import numpy as np
import torch

from deps.neural_persistence.src.tda import PerLayerCalculation


def prepare_ordered_dict_from_model(model):
    """
    tda.PerLayerCalculation requires the weights of a model to be presented as an OrderedDict.

    :param model: An Ordered Dict containing weights and biases.
    :return: An OrderedDict containing layer_name:numpy_weights tuples.
    """

    weights = OrderedDict()
    weight_keys = [key for key in model.keys() if "weight" in key]  # filter biases
    for key in weight_keys:
        weights.update({key: model[key].detach().numpy()})
    return weights


def load_masked_weights(model_path, mask_path):
    """
    Loads a masked network and returns an OrderedDict of the format {layer:masked weights} for use with the neural
    persistence calculations.


    :param model_path: string, path to the end_model
    :param mask_path: string, path to the mask
    :return: OrderedDict
    """

    model = torch.load(model_path)
    mask = torch.load(mask_path)

    for name, param in model.items():
        if name in mask.keys():
            param.data *= mask[name]

    return prepare_ordered_dict_from_model(model)


def get_paths_from_replicate(base_path, experiment_type, eps):
    """
    Creates the paths of all logger files, sparsity reports and relevant models (model_ep0_it0.pth and
    model_epN_it0.pth) from a given replicate directory of a OpenLTH style experiment.
    This is dependant on the type of the experiment (since the folder structure below the root folder is different
    depending on the experiment performed.)

    :param base_path: str, path to the parent directory, e.g.
    "experiments/train_574e51abc295d8da78175b320504f2ba/replicate_2/"
    :param experiment_type: str, one of "train", "lottery", "lottery_branch" depending on the lottery ticket experiment
    you want to analyze
    :param eps: int, number of training epochs. Necessary to generate the path to the trained model.
    :return: dict, with the types of paths (logger, model_start, etc.) as keys and a string (for train) and a list of
    strings (for lottery) as values.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError("The path '{}' is not a valid path.".format(base_path))

    if not os.path.isdir(base_path):
        raise NotADirectoryError("The path '{}' does not lead to a directory. Please provide a directory."
                                 .format(base_path))

    if base_path[-1] != "/":
        base_path = base_path + "/"

    if experiment_type not in base_path:
        raise ValueError("You provided the experiment type {}, but the path suggests another type ({}). Please add "
                         "the experiment type to the path, if you created your own filenames during the OpenLTH "
                         "experiments.".format(experiment_type, base_path))
    # special case because I don't want to compile a regex just for the error of experiment type "lottery" as type on a
    # "lottery_branch" experiment.
    if experiment_type == "lottery" and "lottery_branch" in base_path:
        raise ValueError("You provided the experiment type {}, but the path suggests another type ({}). Please add "
                         "the experiment type to the path, if you created your own filenames during the OpenLTH "
                         "experiments.".format(experiment_type, base_path))

    if experiment_type == "train":
        base_path = base_path + "main/"
        file_paths = {"accuracy": base_path + "logger", "model_start": base_path + "model_ep0_it0.pth",
                      "model_end": base_path + "model_ep{}_it0.pth".format(str(eps))}

    elif experiment_type == "lottery":
        file_paths = {"accuracy": [], "sparsity": [], "model_start": [], "model_end": []}
        levels = os.listdir(base_path)
        for level in levels:
            file_paths["accuracy"].append(base_path + level + "/main/logger")
            file_paths["sparsity"].append(base_path + level + "/main/sparsity_report.json")
            file_paths["model_start"].append(base_path + level + "/main/model_ep0_it0.pth")
            if level == "level_0":
                file_paths["model_end"].append((base_path + level + "/main/model_ep{}_it0.pth".format(str(eps)), None))
            else:
                file_paths["model_end"].append((base_path + level + "/main/model_ep{}_it0.pth".format(str(eps)),
                                                base_path + level + "/main/mask.pth"))

    elif experiment_type == "lottery_branch":
        raise NotImplementedError("Lottery_branch functionality not implemented, yet")
    else:
        raise ValueError("Unknown option '{}' for experiment type. Please choose from "
                         "['train', 'lottery', 'lottery_branch'].".format(experiment_type))

    for file_path in file_paths.values():
        if type(file_path) == str:
            if not os.path.exists(file_path):
                raise FileNotFoundError("The path {} does not point to a file. Please check the parameters you "
                                        "provided. Path: {}, experiment type: {}, epochs: {}"
                                        .format(file_path, base_path, experiment_type, eps))
        elif type(file_path) == list:
            for fp in file_path:
                if type(fp) == str:
                    if not os.path.exists(fp):
                        raise FileNotFoundError("The path {} does not point to a file. Please check the parameters you "
                                                "provided. Path: {}, experiment type: {}, epochs: {}"
                                                .format(fp, base_path, experiment_type, eps))
                elif type(fp) == tuple:
                    if not os.path.exists(fp[0]):
                        raise FileNotFoundError("The path {} does not point to a file. Please check the parameters you "
                                                "provided. Path: {}, experiment type: {}, epochs: {}"
                                                .format(fp[0], base_path, experiment_type, eps))
                    if fp[1] is not None:
                        if not os.path.exists(fp[1]):
                            raise FileNotFoundError("The path {} does not point to a file. Please check the parameters "
                                                    "you provided. Path: {}, experiment type: {}, epochs: {}"
                                                    .format(fp[1], base_path, experiment_type, eps))
                else:
                    raise TypeError("The type of fp ({}) is neither a string nor a tuple. Something went wrong when "
                                    "creating the model_end tuples (paths to model at the end of training and mask)."
                                    .format(type(fp)))
        else:
            raise TypeError("The type of file_paths.values() ({}) is neither string nor list. Something went wrong "
                            "when creating the file paths.".format(type(file_paths.values())))
    return file_paths


def get_paths_from_experiment(base_path, experiment_type, eps):
    """
    Works on top of get_paths_from_replicate to generate all paths for a series of OpenLTH experiments (read: for all
    replicates of an experiemnt) from the experiment root directory.

    :param base_path: str, path of the OpenLTH experiment to generate all paths for
    :param experiment_type: str, type of the experiment in ["lottery", "lottery_branch", "train"]
    :param eps: int > 0, number of training epochs
    :return: dict of dicts of format {replicate_N:{paths as dicts as returned by get_paths_from_replicate}
    """

    if not os.path.exists(base_path):
        raise FileNotFoundError("The path {} does not exist.".format(base_path))

    if base_path[-1] != "/":
        base_path = base_path + "/"

    paths = {}

    # todo once all experiment types are implemented, we can refactor the if-elif-else construction to just if (in valid types)-else and just pass the types
    if experiment_type == "train" or experiment_type == "lottery_branch":
        raise NotImplementedError
    elif experiment_type == "lottery":
        # filtering out directories for plotting and other non-replicate directories -- quick-and-dirty version of the
        # test without re
        replicates = [path for path in os.listdir(base_path) if "replicate" in path]
        if len(replicates) == 0:
            raise FileNotFoundError("The directory {} contains no replicate subdirectories.".format(base_path))
        for replicate in replicates:
            paths.update({replicate: get_paths_from_replicate(base_path + replicate, experiment_type, eps)})
    else:
        raise ValueError("The type {} is not a valid experiment type. Possible types are 'lottery', 'lottery_branch' "
                         "and 'train'.".format(experiment_type))

    # todo simplify when refactoring
    expected_lengths = {key: len(paths["replicate_1"][key]) for key in paths["replicate_1"].keys()}
    for replicate in paths.keys():
        for key in expected_lengths.keys():
            assert len(paths[replicate][key]) == expected_lengths[key], \
                "The number of pruning levels in replicate {} differs from the number of pruning levels in the first " \
                "replicate. Error found in the extraction of these paths: {}: {} and replicate_1: {}."\
                .format(replicate, replicate, paths[replicate][key], paths["replicate_1"][key])
    return paths


def load_unmasked_weights(path):
    """
    Loads unmasked weights from file.

    Implicit assumption: all params in model.named_parameters() that contain the string 'weight' are considered weights
    and loaded.

    :param path: str, path to the file containing the unmasked weights.
    :return: OrderedDict containing name: parameter key-value pairs
    """
    model = torch.load(path)
    weights = OrderedDict()
    for name, param in model.items():
        if "weight" in name:
            weights.update({name: param.detach().numpy()})
    return weights


def load_accuracy(path):
    """
    Loads the test accuracy at the end of training, given the path to a logger file.

    :param path: string, path to logger
    :return: float, accuracy
    """
    with open(path) as file:
        lines = file.readlines()
        accuracy_line = lines[-2]
        # asserts to ensure that mistakes in the logger are found.
        assert "test_accuracy" in accuracy_line, \
            "The line {} does not correspond to the test_accuracy.".format(accuracy_line)
        accuracy = float(accuracy_line.split(",")[-1].strip())
        assert 0. <= accuracy <= 1., "The accuracy {} is not in [0., 1.]."
        return accuracy


def load_sparsity(path):
    """
    Computes the sparsity from a json file containing the number of the total weights and the unpruned weights.
    Expected keywords are "unpruned" and "total".

    :param path: string, path to json file containing the number of total and unpruned weights.
    :return: float, sparsity
    """
    with open(path) as file:
        report = json.load(file)
        sparsity = report['unpruned']/report['total']
        assert 0. <= sparsity <= 1., "The sparsity {} is outside the possible interval [0, 1]. Please check the " \
                                     "corresponding sparsity report {}.".format(sparsity, path)
        return sparsity


def prepare_neural_persistence_for_plotting(neural_persistences):
    """
    Prepares neural persistence outputs for convenient plotting wih matplotlib. See parameters and return for details.

    :param neural_persistences: list of dicts obtained from tda.PerLayerCalculation
    :return: dict of lists with keys being the layer names and an additional key global, the values being lists of
    floats representing the averaged neural persistences
    """
    neural_pers_for_plotting = {key: [] for key in neural_persistences[0].keys()}

    for neural_pers in neural_persistences:
        for key, value in neural_pers.items():
            if key == "global":
                neural_pers_for_plotting["global"].append(value["accumulated_total_persistence_normalized"])
            else:
                neural_pers_for_plotting[key].append(value["total_persistence_normalized"])
    return neural_pers_for_plotting


def load_sparsities_of_replicate(paths):
    sparsities = []
    for report_path in paths["sparsity"]:
        sparsities.append(load_sparsity(report_path))
    return sparsities


def load_accuracies_of_replicate(paths):
    accuracies = []
    for logger_path in paths["accuracy"]:
        accuracies.append(load_accuracy(logger_path))
    return accuracies


def load_neural_persistences_of_replicate(paths):
    neural_persistences = []

    for end_model_path, mask_path in paths["model_end"]:
        if mask_path is None:
            neural_persistences.append(get_neural_persistence_for_unmasked_weights(end_model_path))
        else:
            neural_persistences.append(get_neural_persistence_for_masked_weights(end_model_path, mask_path))

    return neural_persistences


def get_neural_persistence_for_unmasked_weights(model_path):
    neural_pers_calc = PerLayerCalculation()
    return neural_pers_calc(load_unmasked_weights(model_path))


def get_neural_persistence_for_masked_weights(model_path, mask_path):
    neural_pers_calc = PerLayerCalculation()
    return neural_pers_calc(load_masked_weights(model_path, mask_path))


def load_sparsities_of_experiment(paths):
    for i, replicate in enumerate(paths.keys()):
        if i == 0:
            first_replicate = replicate
            sparsities = [load_sparsity(spars) for spars in paths[replicate]["sparsity"]]
        else:
            sparsities_to_be_checked = [load_sparsity(spars) for spars in paths[replicate]["sparsity"]]
            assert_sparsities_equal(sparsities, sparsities_to_be_checked, first_replicate, replicate)
    return sparsities


def assert_sparsities_equal(expected_sparsities, actual_sparsities, key_expected_sparsities, key_actual_sparsities):
    assert actual_sparsities == expected_sparsities, \
        "The sparsities in replicate {} differ from the sparsities in replicate {}, although they should " \
        "be equal. The sparsities in question are {} and {} respectively. Please make sure that you did " \
        "not mix up any experiments.".format(key_expected_sparsities, key_actual_sparsities,
                                             expected_sparsities, actual_sparsities)


def load_accuracies_of_experiment(paths):
    for i, replicate in enumerate(paths.keys()):
        if i == 0:
            accuracies = np.ones((len(paths.keys()), len(paths[replicate]["accuracy"]))) * (-1)
        accuracies[i] = [load_accuracy(acc) for acc in paths[replicate]["accuracy"]]
    return accuracies


def load_neural_persistences_of_experiment(paths):
    # format:
    # [[replicate1.sparsity_level0, replicate1.sparsity_level1, ...],
    #  [replicate2.sparsity_level0, replicate2.sparsity_level1, ...],
    #  ...]
    # where each replicateX.sparsity_levelY is a dict containing the neural persistences
    neural_persistences = []
    for replicate in paths.keys():
        np_for_replicate = []
        for end_model_path, mask_path in paths[replicate]["model_end"]:
            if mask_path is None:
                np_for_replicate.append(get_neural_persistence_for_unmasked_weights(end_model_path))
            else:
                np_for_replicate.append(get_neural_persistence_for_masked_weights(end_model_path, mask_path))
        neural_persistences.append(np_for_replicate)
    return neural_persistences
