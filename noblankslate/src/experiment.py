import os.path

import matplotlib.pyplot as plt

from deps.neural_persistence.src.tda import PerLayerCalculation
import src.utils as utils


# todo merge last commit with next commit in git


# plots:
    # * pruning rate-accuracy as sanity check
    # * pruning rate-NP (variant 1: NP global, variant 2: NP layerwise; potentially solve plotting them as 2 or as 1
    # plot as flag (for networks with more layers; option: add layer grouping?)
    # * accuracy-NP as a sanity check in a way (do both global and layer-wise. potentially put this into different
    # function.)
    # * NP of mask vs NP of masked weights


def sparsity_accuracy_plot(experiment_root_path, eps, show_plot=True, save_plot=False):
    """
    Creates a sparsity-accuracy plot given a "lottery" type LTH experiment. May be extended to "lottery_branch"
    experiments in the future.
    This is pretty standard for LTH work and mainly is meant as a sanity check.

    :param experiment_root_path: string. Root path of a "lottery" type experiment.
    :param eps: int, number of training epochs
    :param show_plot: bool, default: True, controls showing of plot
    :param save_plot: bool, default: False, controls saving of plot. The plot will be saved to
    experiment_root_path/plots/sporsity-accuracy.png
    :return: tuple containing the list of sparsities and the list of accuracies. This might be changed in the future
    to work with TikZ.
    """
    paths = utils.get_file_paths(experiment_root_path, "lottery", eps)

    accuracies = []
    sparsities = []

    for logger_path in paths["logger"]:
        accuracies.append(utils.load_accuracy(logger_path))
    for report_path in paths["sparsity_report"]:
        sparsities.append(utils.load_sparsity(report_path))

    _, p = plt.subplots(1, 1)
    p.plot(sparsities, accuracies)
    p.invert_xaxis()

    if save_plot:
        if experiment_root_path[-1] != "/":
            experiment_root_path += "/"
        if not os.path.isdir(experiment_root_path + "plots/"):
            os.mkdir(experiment_root_path + "plots/")
        plt.savefig(experiment_root_path + "plots/sparsity_accuracy.png")
    # since plt.show() clears the current figure, saving first and then showing avoids running into problems.
    if show_plot:
        plt.show()

    return sparsities, accuracies


def sparsity_neural_persistence_plot(experiment_root_path, eps, show_plot=True, save_plot=False):
    """
    Creates sparsity-neural persistence plots.

    :param experiment_root_path: str, path to root directory of a "lottery" type experiment.
    :param eps: int, number of training epochs.
    :param show_plot: bool, default: True
    :param save_plot: bool, default: False, plot will be saved to
    experiment_root_path/plots/sparsity_neural_persistence.png
    :return: tuple containing the list of sparsities and a list containing the output of tda.PerLayerCalculation() for
    each pruned network.
    """
    paths = utils.get_file_paths(experiment_root_path, "lottery", eps)

    sparsities = []
    for report in paths["sparsity_report"]:
        sparsities.append(utils.load_sparsity(report))

    neural_persistences = []
    neural_pers_calc = PerLayerCalculation()

    for end_model_path, mask_path in paths["model_end"]:
        if mask_path is None:
            neural_persistences.append(neural_pers_calc(utils.load_unmasked_weights(end_model_path)))
        else:
            neural_persistences.append(neural_pers_calc(utils.load_masked_weights(end_model_path, mask_path)))

    # todo this will become it's own utiliy function once NP is used again in a plotter.
    neural_pers_for_plotting = {key: [] for key in neural_persistences[0].keys()}

    for neural_pers in neural_persistences:
        print(neural_pers)
        for key, value in neural_pers.items():
            if key == "global":
                neural_pers_for_plotting["global"].append(value["accumulated_total_persistence_normalized"])
            else:
                neural_pers_for_plotting[key].append(value["total_persistence_normalized"])

    _, p = plt.subplots(1, 1)

    for key, np_plot in neural_pers_for_plotting.items():
        p.plot(sparsities, np_plot, label=key)
    p.invert_xaxis()
    p.legend()

    if save_plot:
        if experiment_root_path[-1] != "/":
            experiment_root_path += "/"
        if not os.path.isdir(experiment_root_path + "plots/"):
            os.mkdir(experiment_root_path + "plots/")
        plt.savefig(experiment_root_path + "plots/sparsity_neural_persistence.png")
    # since plt.show() clears the current figure, saving first and then showing avoids running into problems.
    if show_plot:
        plt.show()

    return sparsities, neural_persistences


# assumption: the open LTH side of the experiment is already run
# works with "lottery" experiment type for now
# answers base question: is there any difference wrt NP in pruned networks depending on the level of sparsity?
# plots: pruning rate vs NP, pruning rate vs accuracy, NP vs accuracy (is there any correlation?)


def accuracy_neural_persistence(experiment_root_path, eps):
    """"""
    paths = utils.get_file_paths(experiment_root_path, "lottery", eps)

    print(paths)
    return 0
