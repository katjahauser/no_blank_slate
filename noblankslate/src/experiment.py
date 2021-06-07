import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deps.neural_persistence.src.tda import PerLayerCalculation
import src.utils as utils


# plots:
# * NP of mask vs NP of masked weights
# todo add replicates to replicate plotting function save paths

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


def sparsity_accuracy_plot_experiment(root_path, eps, show_plot=True, save_plot=False):
    """
    Create the sparsity-accuracy plot corresponding to a openLTH experiment with several replicates.

    The mean over the accuracies is plotted including the std. deviation as error bars. It is asserted that the
    sparsities of all replicates are the same to detect errors from mixing things up.

    :param root_path: experiment root directory, i.e. the directory in which the replicate live and in which the plots
    directory will be created.
    :param eps: number of training epochs
    :param show_plot: bool, default: True, shows plot
    :param save_plot: bool, default: False, saves plot to root_path/plots/sparsity_accuracy_experiment.png
    :return: three lists of floats, the first for sparsities, the second for mean accuracies at each pruning step, the
    last for the standard deviation at each pruning step.
    """

    paths = utils.get_paths_from_experiment(root_path, "lottery", eps)
    accuracies = np.ones((len(paths.keys()), len(paths["replicate_1"]["accuracy"]))) * (-1)
    for i, replicate in enumerate(paths.keys()):
        accuracies[i] = [utils.load_accuracy(acc) for acc in paths[replicate]["accuracy"]]
    means = list(np.mean(accuracies, axis=0))
    stds = list(np.std(accuracies, axis=0))

    sparsities = [utils.load_sparsity(spars) for spars in paths["replicate_1"]["sparsity"]]

    for replicate in paths.keys():
        if replicate != "replicate_1":
            sparsities_to_be_checked = [utils.load_sparsity(spars) for spars in paths[replicate]["sparsity"]]
            assert sparsities_to_be_checked == sparsities, \
                "The sparsities in replicate {} differ from the sparsities in the first replicate. {} and {} " \
                "respectively. Make sure you did not mix any experiments.".format(replicate, sparsities_to_be_checked,
                                                                                  sparsities)

    _, p = plt.subplots(1, 1)
    p.errorbar(sparsities, means, stds)
    p.invert_xaxis()
    plt.title("Sparsity-Accuracy over {} runs".format(len(paths.keys())))
    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy")

    if save_plot:
        if root_path[-1] != "/":
            root_path = root_path + "/"
        if not os.path.isdir(root_path + "plots/"):
            os.mkdir(root_path + "plots/")
        plt.savefig(root_path + "plots/sparsity_accuracy_experiment.png")
    # since plt.show() clears the current figure, saving first and then showing avoids running into problems.
    if show_plot:
        plt.show()

    return sparsities, means, stds


def sparsity_neural_persistence_plot_experiment(root_path, eps, show_plot=True, save_plot=False):
    """
    Creates sparsity-neural persistence plots for an openLTH experiment with several replicates.
    Sanity check for making sure that all experiments share the same sparsity level are included.

    :param root_path: experiment root directory, i.e. the directory in which the replicate live and in which the plots
    directory will be created.
    :param eps: number of training epochs
    :param show_plot: bool, default: True, shows plot
    :param save_plot: bool, default: False, saves plot to root_path/plots/sparsity_neural_persistence_experiment.png
    :return: three lists, the first list is a list of floats (sparsities), the second and third are lists of dicts
    in the format {layer_name: mean_normalized_persistence, ..., global:mean_normalized_global_persistence} and
    {layer_name: std_dev_normalized_persistence, ..., global:std_dev_normalized_global_persistence} respectively.
    """

    paths = utils.get_paths_from_experiment(root_path, "lottery", eps)

    sparsities = [utils.load_sparsity(spars) for spars in paths["replicate_1"]["sparsity"]]

    for replicate in paths.keys():
        if replicate != "replicate_1":
            sparsities_to_be_checked = [utils.load_sparsity(spars) for spars in paths[replicate]["sparsity"]]
            assert sparsities_to_be_checked == sparsities, \
                "The sparsities in replicate {} differ from the sparsities in the first replicate. {} and {} " \
                "respectively. Make sure you did not mix any experiments.".format(replicate, sparsities_to_be_checked,
                                                                                  sparsities)

    neural_persistences_raw = [[] for _ in range(len(sparsities))]
    neural_pers_calc = PerLayerCalculation()

    for replicate in paths.keys():
        for i, (end_model_path, mask_path) in enumerate(paths[replicate]["model_end"]):
            if mask_path is None:
                neural_persistences_raw[i].append(neural_pers_calc(utils.load_unmasked_weights(end_model_path)))
            else:
                neural_persistences_raw[i].append(neural_pers_calc(utils.load_masked_weights(end_model_path, mask_path)))

    mean_neural_persistences = []
    std_neural_persistences = []
    means_for_plotting = [[] for _ in range(len(neural_persistences_raw[0][0].keys()))]
    stds_for_plotting = [[] for _ in range(len(neural_persistences_raw[0][0].keys()))]

    for neural_persistences_in_one_level in neural_persistences_raw:
        neural_persistence_array = np.zeros((len(neural_persistences_raw[0]), len(neural_persistences_raw[0][0].keys())))
        for i, neural_pers in enumerate(neural_persistences_in_one_level):
            for j, layer_key in enumerate(neural_pers.keys()):
                for persistence_key in neural_pers[layer_key].keys():
                    if "normalized" in persistence_key:
                        neural_persistence_array[i, j] = neural_pers[layer_key][persistence_key]

        mean_nps = np.mean(neural_persistence_array, axis=0)
        std_nps = np.std(neural_persistence_array, axis=0)

        mean_neural_persistences.append({key: mean_nps[i] for i, key in enumerate(neural_persistences_raw[0][0].keys())})
        std_neural_persistences.append({key: std_nps[i] for i, key in enumerate(neural_persistences_raw[0][0].keys())})
        for i in range(len(mean_nps)):
            means_for_plotting[i].append(mean_nps[i])
            stds_for_plotting[i].append(std_nps[i])

    _, p = plt.subplots(1, 1)
    for i in range(len(means_for_plotting)):
        p.errorbar(sparsities, means_for_plotting[i], stds_for_plotting[i])
    p.invert_xaxis()
    plt.title("Sparsity-Neural Persistence over {} runs".format(len(paths.keys())))
    plt.xlabel("Sparsity")
    plt.ylabel("Neural Persistence")

    if save_plot:
        if root_path[-1] != "/":
            root_path = root_path + "/"
        if not os.path.isdir(root_path + "plots/"):
            os.mkdir(root_path + "plots/")
        plt.savefig(root_path + "plots/sparsity_neural_persistence_experiment.png")
    # since plt.show() clears the current figure, saving first and then showing avoids running into problems.
    if show_plot:
        plt.show()

    return sparsities, mean_neural_persistences, std_neural_persistences


def accuracy_neural_persistence_plot_experiment(experiment_root_path, eps, show_plot=True, save_plot=False):
    """
    Creates accuracy-neural persistence plots for experiments.

    todo as with the same function for one replicate: maybe add option to plot global in separate plot

    :param experiment_root_path: str, root path of an openLTH lottery-type experiment
    :param eps: training epochs
    :param show_plot: bool, default=True
    :param save_plot: bool, default=False, saves plot to
    experiment_root_path/plots/accuracy_neural_persistence_experiment.png
    :return: two lists, first list of accuracies, second list of dicts as put out by tda.PerLayerCalc. list1[i]
    corresponds to list2[i] for convenient use with matplotlib.pyplot.scatter.
    """
    paths = utils.get_paths_from_experiment(experiment_root_path, "lottery", eps)
    accuracies = []
    neural_pers_calc = PerLayerCalculation()
    neural_persistences = []

    for replicate in paths.keys():
        for accuracy_path in paths[replicate]["accuracy"]:
            accuracies.append(utils.load_accuracy(accuracy_path))
        for end_model_path in paths[replicate]["model_end"]:
            if end_model_path[1] is None:
                neural_persistences.append(neural_pers_calc(utils.load_unmasked_weights(end_model_path[0])))
            else:
                neural_persistences.append(neural_pers_calc(utils.load_masked_weights(end_model_path[0],
                                                                                      end_model_path[1])))

    neural_pers_for_plotting = utils.prepare_neural_persistence_for_plotting(neural_persistences)

    _, p = plt.subplots(1, 1)

    # todo this needs to be refactored. Add tests and asserts (for both markers <= num_replicate and colors <= num_layers+1) then, too.
    markers = "ov2P*x+Ds<3p"
    colors = "rgbcmy"
    for i, (key, np_plot) in enumerate(neural_pers_for_plotting.items()):
        for j, n_pers in enumerate(np_plot):
            p.scatter(accuracies[j], n_pers, marker=markers[int(j/len(paths.keys()))], color=colors[i])
    colors_for_legend = [plt.Circle((0, 0), color=colors[i]) for i in range(len(neural_pers_for_plotting.keys()))]
    p.legend(colors_for_legend, neural_pers_for_plotting.keys())
    plt.title("Accuracy-Neural Persistence")
    plt.xlabel("Accuracy")
    plt.ylabel("Neural Persistence")

    if save_plot:
        if experiment_root_path[-1] != "/":
            experiment_root_path += "/"
        plot_dir = experiment_root_path + "/plots/"
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        plt.savefig(plot_dir + "/accuracy_neural_persistence_experiment.png")
    # since plt.show() clears the current figure, saving first and then showing avoids running into problems.
    if show_plot:
        plt.show()

    return accuracies, neural_persistences


if __name__ == "__main__":
    sparsity_accuracy_plot_experiment("../experiments/lottery_37adeb06fd584c18ebbf48beec5747d3/", 20, save_plot=True)
    sparsity_neural_persistence_plot_experiment("../experiments/lottery_37adeb06fd584c18ebbf48beec5747d3/", 20, save_plot=True)
    accuracy_neural_persistence_plot_experiment("../experiments/lottery_37adeb06fd584c18ebbf48beec5747d3/", 20, save_plot=True)
