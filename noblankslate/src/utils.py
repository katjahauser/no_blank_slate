from collections import OrderedDict

import numpy as np


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


    # todo in utils einpflegen
    # todo tests schreiben
def load_masked_network(path, num_layers):
    """
    Loads a masked network from a given path and returns an OrderedDict of the format {layer:masked weights} for
    use with the neural persistence calculations.

    Internally, both the network and the mask are loaded, then the mask is applied. This function implicitly
    assumes that the format of the open lottery ticket hypothesis framework is used.

    :param path: string
    :return: OrderedDict
    """

    # todo checks: path is a directory not a file
    # todo check: allow alternative naming for mask and weights folder and check if they exist

    masked_weights = OrderedDict()

    for i in range(num_layers):
        final_weights = np.load(path+"final/layer{}.npy".format(str(i)))
    # todo handling for missing masks in first training step (+ adaption of description/docu)
        try:
            mask = np.load(path+"masks/layer{}.npy".format(str(i)))
            masked_weights["layer{}".format(str(i))] = final_weights * mask
        except FileNotFoundError:
            print("No mask found at path {}. Ignore this, if this is the "
                  "first (i.e., unpruned) network. In the LTH framework, this is folder 0. Returning only the "
                  "weights.".format(path + "masks/layer{}.npy".format(str(i))))
            masked_weights["layer{}".format(str(i))] = final_weights

    return masked_weights


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