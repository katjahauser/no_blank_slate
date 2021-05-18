from collections import OrderedDict
import os.path

import numpy as np
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
        weights.update({key: model.state_dict()[key]})
    return weights


def load_masked_weights(path, num_eps):
    """
    Loads a masked network from a given directory path and returns an OrderedDict of the format {layer:masked weights}
    for use with the neural persistence calculations.

    This function implicitly assumes that the format of the open lottery ticket hypothesis framework is used. This
    means, that the mask and the trained network live in the directory. The mask as mask.pth, the model as
    model_epX_it0.pth when trained for X episodes.

    :param path: string, path to the directory the files live in.
    :param num_eps: int, number of training epochs.
    :return: OrderedDict
    """

    if path[-1] != "/":
        path = path + "/"

    model_path = path + "model_ep{}_it0.pth".format(str(num_eps))
    mask_path = path + "mask.pth"

    model = torch.load(model_path)
    mask = torch.load(mask_path)

    for name, param in model.named_parameters():
        if name in mask.keys():
            param.data *= mask[name]

    return prepare_ordered_dict_from_model(model)


def get_file_paths(path, experiment_type, eps):
    """
    Loads the paths of all log (logger) files and relevant models (model_ep0_it0.pth and model_epN_it0.pth) from a
    given root directory of a OpenLTH style experiment based on the type of the experiment (since the folder structure
    is different depending on the experiment performed.)
    :param path: str, path to the parent directory, e.g. "experiments/train_574e51abc295d8da78175b320504f2ba/"
    :param experiment_type: str, one of "train", "lottery", "lottery_branch" depending on the lottery ticket experiment
    you want to analyze
    :param eps: int, number of training epochs. Necessary to generate the path to the trained model.
    :return: list of strings (the paths)
    """
    if not os.path.exists(path):
        raise FileNotFoundError("The path '{}' is not a valid path.".format(path))

    if not os.path.isdir(path):
        raise NotADirectoryError("The path '{}' does not lead to a directory. Please provide a directory.".format(path))

    if path[-1] != "/":
        path = path + "/"

    file_paths = []

    if experiment_type not in path:
        raise ValueError("You provided the experiment type {}, but the path suggests another type ({}). Please add "
                         "the experiment type to the path, if you created your own filenames during the OpenLTH "
                         "experiments.".format(experiment_type, path))
    # special case because I don't want to compile a regex just for the error of experiment type "lottery" as type on a
    # "lottery_branch" experiment.
    if experiment_type == "lottery" and "lottery_branch" in path:
        raise ValueError("You provided the experiment type {}, but the path suggests another type ({}). Please add "
                         "the experiment type to the path, if you created your own filenames during the OpenLTH "
                         "experiments.".format(experiment_type, path))

    if experiment_type == "train":
        base_path = path + "replicate_1/main/"
        file_paths.extend([base_path + "logger",  base_path + "model_ep0_it0.pth",
                           base_path + "model_ep{}_it0.pth".format(str(eps))])

    elif experiment_type == "lottery":
        levels = os.listdir(path + "replicate_1/")
        base_path = path + "replicate_1/"
        for level in levels:
            file_paths.extend([base_path + level + "/main/logger", base_path + level + "/main/model_ep0_it0.pth",
                               base_path + level + "/main/model_ep{}_it0.pth".format(str(eps))])

    elif experiment_type == "lottery_branch":
        raise NotImplementedError("Lottery_branch functionality not implemented, yet")
    else:
        raise ValueError("Unknown option '{}' for experiment type. Please choose from "
                         "['train', 'lottery', 'lottery_branch'].".format(experiment_type))

    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError("The path {} does not point to a file. Please check the parameters you provided. "
                                    "Path: {}, experiment type: {}, epochs: {}".format(file_path, path,
                                                                                       experiment_type, eps))
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
            weights.update({name: param.detach()})
    return weights

# todo implement and load_accuracy, implement "main experiment function"
