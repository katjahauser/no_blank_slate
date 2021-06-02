import os.path
from pathlib import Path

import matplotlib.pyplot as plt

from deps.neural_persistence.src.tda import PerLayerCalculation
import src.utils as utils


# plots:
# * NP of mask vs NP of masked weights

def sparsity_accuracy_plot_replicate(experiment_root_path, eps, show_plot=True, save_plot=False):
    """
    Creates a sparsity-accuracy plot given a "lottery" type LTH experiment. May be extended to "lottery_branch"
    experiments in the future.
    This is a standard plot for LTH work and mainly is meant as a sanity check.

    :param experiment_root_path: string. Root path of a "lottery" type experiment.
    :param eps: int, number of training epochs
    :param show_plot: bool, default: True, controls showing of plot
    :param save_plot: bool, default: False, controls saving of plot. The plot will be saved to
    experiment_root_path/plots/sporsity-accuracy.png
    :return: tuple containing the list of sparsities and the list of accuracies. This might be changed in the future
    to work with TikZ.
    """
    paths = utils.get_paths_from_replicate(experiment_root_path, "lottery", eps)

    accuracies = []
    sparsities = []

    for logger_path in paths["accuracy"]:
        accuracies.append(utils.load_accuracy(logger_path))
    for report_path in paths["sparsity"]:
        sparsities.append(utils.load_sparsity(report_path))

    _, p = plt.subplots(1, 1)
    p.plot(sparsities, accuracies)
    p.invert_xaxis()
    plt.title("Sparsity-Accuracy")
    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy")

    if save_plot:
        plot_dir = str(Path(experiment_root_path).parent) + "/plots/"
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        plt.savefig(plot_dir + "sparsity_accuracy.png")
    # since plt.show() clears the current figure, saving first and then showing avoids running into problems.
    if show_plot:
        plt.show()

    return sparsities, accuracies


def sparsity_neural_persistence_plot_replicate(experiment_root_path, eps, show_plot=True, save_plot=False):
    """
    Creates sparsity-neural persistence plots.

    todo add option to plot layers in one and global in another plot? might be better arrangement for models with a large number of layers.

    :param experiment_root_path: str, path to root directory of a "lottery" type experiment.
    :param eps: int, number of training epochs.
    :param show_plot: bool, default: True
    :param save_plot: bool, default: False, plot will be saved to
    experiment_root_path/plots/sparsity_neural_persistence.png
    :return: tuple containing the list of sparsities and a list containing the output of tda.PerLayerCalculation() for
    each pruned network.
    """
    paths = utils.get_paths_from_replicate(experiment_root_path, "lottery", eps)

    sparsities = []
    for report in paths["sparsity"]:
        sparsities.append(utils.load_sparsity(report))

    neural_persistences = []
    neural_pers_calc = PerLayerCalculation()

    for end_model_path, mask_path in paths["model_end"]:
        if mask_path is None:
            neural_persistences.append(neural_pers_calc(utils.load_unmasked_weights(end_model_path)))
        else:
            neural_persistences.append(neural_pers_calc(utils.load_masked_weights(end_model_path, mask_path)))

    neural_pers_for_plotting = utils.prepare_neural_persistence_for_plotting(neural_persistences)

    _, p = plt.subplots(1, 1)

    for key, np_plot in neural_pers_for_plotting.items():
        p.plot(sparsities, np_plot, label=key)
    p.invert_xaxis()
    p.legend()
    plt.title("Sparsity-Neural Persistence")
    plt.xlabel("Sparsity")
    plt.ylabel("Neural Persistence")

    if save_plot:
        plot_dir = str(Path(experiment_root_path).parent) + "/plots/"
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        plt.savefig(plot_dir + "sparsity_neural_persistence.png")
    # since plt.show() clears the current figure, saving first and then showing avoids running into problems.
    if show_plot:
        plt.show()

    return sparsities, neural_persistences


def accuracy_neural_persistence_plot_replicate(experiment_root_path, eps, show_plot=True, save_plot=False):
    """
    Creates accuracy-neural persistence plots.

    todo add option to plot layers in one and global in another plot? might be better arrangement for models with a large number of layers.

    :param experiment_root_path: string, path to "lottery" type OpenLTH experiment directory
    :param eps: int, number of training epochs
    :param show_plot: bool, default:True
    :param save_plot: bool, default:False, saves plot to experiment_root_path/plots/accuracy_neural_persistence.png
    :return: tuple containing list of accuracies and list of dicts containing the neural persistences as calculated by
    tda.PerLayerCalculation
    """
    paths = utils.get_paths_from_replicate(experiment_root_path, "lottery", eps)

    accuracies = []
    for acc_path in paths["accuracy"]:
        accuracies.append(utils.load_accuracy(acc_path))

    neural_persistences = []
    neural_pers_calc = PerLayerCalculation()

    for end_model_path, mask_path in paths["model_end"]:
        if mask_path is None:
            neural_persistences.append(neural_pers_calc(utils.load_unmasked_weights(end_model_path)))
        else:
            neural_persistences.append(neural_pers_calc(utils.load_masked_weights(end_model_path, mask_path)))

    neural_pers_for_plotting = utils.prepare_neural_persistence_for_plotting(neural_persistences)

    _, p = plt.subplots(1, 1)

    for key, np_plot in neural_pers_for_plotting.items():
        p.scatter(accuracies, np_plot, label=key)
    p.legend()
    plt.title("Accuracy-Neural Persistence")
    plt.xlabel("Accuracy")
    plt.ylabel("Neural Persistence")

    if save_plot:
        plot_dir = str(Path(experiment_root_path).parent) + "/plots/"
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        plt.savefig(plot_dir + "/accuracy_neural_persistence.png")
    # since plt.show() clears the current figure, saving first and then showing avoids running into problems.
    if show_plot:
        plt.show()

    return accuracies, neural_persistences


if __name__ == "__main__":
    sparsity_accuracy_plot_replicate("../experiments/lottery_37adeb06fd584c18ebbf48beec5747d3/", 20, save_plot=True)
    sparsity_neural_persistence_plot_replicate("../experiments/lottery_37adeb06fd584c18ebbf48beec5747d3/", 20, save_plot=True)
    accuracy_neural_persistence_plot_replicate("../experiments/lottery_37adeb06fd584c18ebbf48beec5747d3/", 20, save_plot=True)
