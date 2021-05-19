from collections import OrderedDict
import os.path

import json
import torch


def prepare_ordered_dict_from_model(model):
    """
    tda.PerLayerCalculation requires the weights of a model to be presented as an OrderedDict. Since the open_LTH
    framework uses Pytorch and the neural persistence framework uses Tensorflow, we need a conversion.

    :param model: A Pytorch model.
    :return: An OrderedDict containing layer_name:numpy_weights tuples.
    """

    weights = OrderedDict()
    weight_keys = [key for key in model.state_dict().keys() if "weight" in key]
    for key in weight_keys:
        weights.update({key: model.state_dict()[key].detach().numpy()})
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

    for name, param in model.named_parameters():
        if name in mask.keys():
            param.data *= mask[name]

    return prepare_ordered_dict_from_model(model)


def get_file_paths(experiment_root_path, experiment_type, eps):
    """
    Creates the paths of all logger files, sparsity reports and relevant models (model_ep0_it0.pth and
    model_epN_it0.pth) from a given root directory of a OpenLTH style experiment.
    This is based on the type of the experiment (since the folder structure below the root folder is different
    depending on the experiment performed.)

    :param experiment_root_path: str, path to the parent directory, e.g.
    "experiments/train_574e51abc295d8da78175b320504f2ba/"
    :param experiment_type: str, one of "train", "lottery", "lottery_branch" depending on the lottery ticket experiment
    you want to analyze
    :param eps: int, number of training epochs. Necessary to generate the path to the trained model.
    :return: dict, with the types of paths (logger, model_start, etc.) as keys and a string (for train) and a list of
    strings (for lottery) as values.
    """
    if not os.path.exists(experiment_root_path):
        raise FileNotFoundError("The path '{}' is not a valid path.".format(experiment_root_path))

    if not os.path.isdir(experiment_root_path):
        raise NotADirectoryError("The path '{}' does not lead to a directory. Please provide a directory."
                                 .format(experiment_root_path))

    if experiment_root_path[-1] != "/":
        experiment_root_path = experiment_root_path + "/"

    if experiment_type not in experiment_root_path:
        raise ValueError("You provided the experiment type {}, but the path suggests another type ({}). Please add "
                         "the experiment type to the path, if you created your own filenames during the OpenLTH "
                         "experiments.".format(experiment_type, experiment_root_path))
    # special case because I don't want to compile a regex just for the error of experiment type "lottery" as type on a
    # "lottery_branch" experiment.
    if experiment_type == "lottery" and "lottery_branch" in experiment_root_path:
        raise ValueError("You provided the experiment type {}, but the path suggests another type ({}). Please add "
                         "the experiment type to the path, if you created your own filenames during the OpenLTH "
                         "experiments.".format(experiment_type, experiment_root_path))

    if experiment_type == "train":
        base_path = experiment_root_path + "replicate_1/main/"
        file_paths = {"logger": base_path + "logger", "model_start": base_path + "model_ep0_it0.pth",
                      "model_end": base_path + "model_ep{}_it0.pth".format(str(eps))}

    elif experiment_type == "lottery":
        file_paths = {"logger": [], "sparsity_report": [], "model_start": [], "model_end": []}
        levels = os.listdir(experiment_root_path + "replicate_1/")
        base_path = experiment_root_path + "replicate_1/"
        for level in levels:
            file_paths["logger"].append(base_path + level + "/main/logger")
            file_paths["sparsity_report"].append(base_path + level + "/main/sparsity_report.json")
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
                                        .format(file_path, experiment_root_path, experiment_type, eps))
        elif type(file_path) == list:
            for fp in file_path:
                if type(fp) == str:
                    if not os.path.exists(fp):
                        raise FileNotFoundError("The path {} does not point to a file. Please check the parameters you "
                                                "provided. Path: {}, experiment type: {}, epochs: {}"
                                                .format(fp, experiment_root_path, experiment_type, eps))
                elif type(fp) == tuple:
                    if not os.path.exists(fp[0]):
                        raise FileNotFoundError("The path {} does not point to a file. Please check the parameters you "
                                                "provided. Path: {}, experiment type: {}, epochs: {}"
                                                .format(fp[0], experiment_root_path, experiment_type, eps))
                    if fp[1] is not None:
                        if not os.path.exists(fp[1]):
                            raise FileNotFoundError("The path {} does not point to a file. Please check the parameters "
                                                    "you provided. Path: {}, experiment type: {}, epochs: {}"
                                                    .format(fp[1], experiment_root_path, experiment_type, eps))
                else:
                    raise TypeError("The type of fp ({}) is neither a string nor a tuple. Something went wrong when "
                                    "creating the model_end tuples (paths to model at the end of training and mask)."
                                    .format(type(fp)))

        else:
            raise TypeError("The type of file_paths.values() ({}) is neither string nor list. Something went wrong "
                            "when creating the file paths.".format(type(file_paths.values())))
    return file_paths


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
    for name, param in model.named_parameters():
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
