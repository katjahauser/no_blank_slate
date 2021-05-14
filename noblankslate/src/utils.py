from collections import OrderedDict

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


def load_masked_network(path, num_eps):
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


# def get_test_accuracy(path):
#     """ todo docu :D im wesentlichen wollen wir die test accuracy bei der letzten iteration haben"""
#     acc = -1
#
#     with open(path, "r") as file:
#         last_line = file.readlines()[-1]
#         acc = last_line.split(",")[-1]
#
#     return acc
#
# calc = PerLayerCalculation()
#
# # todo an passender stelle einpflegen, je nach fragestellung refactoren
#
# y = np.zeros((21, 5))
#
# for i in range(21):
#     persistence = calc(load_masked_network("../initial_experiments/mnist_fc_data/trial1/{}/same_init/"
#                                            .format(str(i)), 3))
#     y[i, 0] = persistence["layer0"]["total_persistence_normalized"]
#     y[i, 1] = persistence["layer1"]["total_persistence_normalized"]
#     y[i, 2] = persistence["layer2"]["total_persistence_normalized"]
#     y[i, 3] = persistence["global"]["accumulated_total_persistence_normalized"]
#     y[i, 4] = get_test_accuracy("../initial_experiments/mnist_fc_data/trial1/{}/same_init/test.log".format(str(i)))
#
# x = np.arange(21)
#
# plt.plot(x, y)
# plt.legend(["layer0", "layer1", "layer2", "global", "accuracy"])
# plt.show()