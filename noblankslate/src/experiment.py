import abc
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deps.neural_persistence.src.tda import PerLayerCalculation
import src.plotters as plotters
import src.utils as utils

# plots:
# * NP of mask vs NP of masked weights
# todo add replicates to replicate plotting function save paths


class ReplicateEvaluator(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_paths') and
                callable(subclass.get_paths) and
                hasattr(subclass, 'load_x_data') and
                callable(subclass.load_x_data) and
                hasattr(subclass, 'load_y_data') and
                callable(subclass.load_y_data) and
                hasattr(subclass, 'prepare_data_for_plotting') and
                callable(subclass.prepare_data_for_plotting) and
                hasattr(subclass, 'get_plotter') and
                callable(subclass.get_plotter))

    def __init__(self, experiment_root_path, eps):
        self.experiment_root_path = experiment_root_path
        self.raise_if_no_valid_epoch(eps)
        self.epochs = eps
        self.x_data = []
        self.y_data = []

    @staticmethod
    def raise_if_no_valid_epoch(epochs):
        if epochs <= 0:
            raise ValueError("There must be more than 0 epochs. You provided {}.".format(str(epochs)))
        if type(epochs) is not int:
            raise ValueError("Epochs must be an integer. You provided {}.".format(str(epochs)))

    def evaluate(self, show_plot, save_plot):
        self.load_data()
        self.prepare_data_for_plotting()
        plotter = self.get_plotter()
        self.generate_plot(plotter)
        if show_plot:
            plotter.show_plot()
        if save_plot:
            plotter.save_plot(self.experiment_root_path)

    def load_data(self):
        paths = self.get_paths()
        self.load_x_data(paths)
        self.load_y_data(paths)

    @abc.abstractmethod
    def get_paths(self):
        raise NotImplementedError("Trying to call get_paths from abstract base class SingleReplicateHandler.")

    @abc.abstractmethod
    def load_x_data(self, paths):
        raise NotImplementedError("Trying to call load_x_data from abstract base class SingleReplicateHandler.")

    @abc.abstractmethod
    def load_y_data(self, paths):
        raise NotImplementedError("Trying to call load_y_data from abstract base class SingleReplicateHandler.")

    @abc.abstractmethod
    def prepare_data_for_plotting(self):
        raise NotImplementedError("Trying to call prepare_data_for_plotting from abstract base class "
                                  "SingleReplicateHandler.")

    @abc.abstractmethod
    def get_plotter(self):
        raise NotImplementedError("Trying to call get_plotter from abstract base class SingleReplicateHandler.")

    def generate_plot(self, plotter):
        plotter.make_plot(self.x_data, self.y_data)


class SparsityAccuracyOnSingleReplicateEvaluator(ReplicateEvaluator):
    def get_paths(self):
        return utils.get_paths_from_replicate(self.experiment_root_path, "lottery", self.epochs)

    def load_x_data(self, paths):
        sparsities = []
        for report_path in paths["sparsity"]:
            sparsities.append(utils.load_sparsity(report_path))
        self.x_data = sparsities

    def load_y_data(self, paths):
        accuracies = []
        for logger_path in paths["accuracy"]:
            accuracies.append(utils.load_accuracy(logger_path))
        self.y_data = accuracies

    def prepare_data_for_plotting(self):
        pass  # nothing to do for this type of plot

    def get_plotter(self):
        return plotters.SparsityAccuracyReplicatePlotter()


class SparsityNeuralPersistenceOnSingleReplicateEvaluator(ReplicateEvaluator):
    def get_paths(self):
        return utils.get_paths_from_replicate(self.experiment_root_path, "lottery", self.epochs)

    def load_x_data(self, paths):
        sparsities = []
        for report_path in paths["sparsity"]:
            sparsities.append(utils.load_sparsity(report_path))
        self.x_data = sparsities

    def load_y_data(self, paths):
        neural_persistences = []

        for end_model_path, mask_path in paths["model_end"]:
            if mask_path is None:
                neural_persistences.append(get_neural_persistence_for_unmasked_weights(end_model_path))
            else:
                neural_persistences.append(get_neural_persistence_for_masked_weights(end_model_path, mask_path))

        self.y_data = neural_persistences

    def prepare_data_for_plotting(self):
        # nothing to do for self.x_data
        self.y_data = utils.prepare_neural_persistence_for_plotting(self.y_data)

    def get_plotter(self):
        return plotters.SparsityNeuralPersistenceReplicatePlotter()


def get_neural_persistence_for_unmasked_weights(model_path):
    neural_pers_calc = PerLayerCalculation()
    return neural_pers_calc(utils.load_unmasked_weights(model_path))


def get_neural_persistence_for_masked_weights(model_path, mask_path):
    neural_pers_calc = PerLayerCalculation()
    return neural_pers_calc(utils.load_masked_weights(model_path, mask_path))


class AccuracyNeuralPersistenceOnSingleReplicateEvaluator(ReplicateEvaluator):
    def get_paths(self):
        return utils.get_paths_from_replicate(self.experiment_root_path, "lottery", self.epochs)

    def load_x_data(self, paths):
        accuracies = []
        for logger_path in paths["accuracy"]:
            accuracies.append(utils.load_accuracy(logger_path))
        self.x_data = accuracies

    def load_y_data(self, paths):
        neural_persistences = []

        for end_model_path, mask_path in paths["model_end"]:
            if mask_path is None:
                neural_persistences.append(get_neural_persistence_for_unmasked_weights(end_model_path))
            else:
                neural_persistences.append(get_neural_persistence_for_masked_weights(end_model_path, mask_path))

        self.y_data = neural_persistences

    def prepare_data_for_plotting(self):
        # nothing to do for self.x_data
        self.y_data = utils.prepare_neural_persistence_for_plotting(self.y_data)

    def get_plotter(self):
        return plotters.AccuracyNeuralPersistenceReplicatePlotter()


class ExperimentEvaluator(ReplicateEvaluator):
    def __init__(self, experiment_root_path, eps):
        super().__init__(experiment_root_path, eps)
        self.num_replicates = self.get_num_replicates()

    def get_num_replicates(self):
        paths = self.get_paths()
        return len(paths.keys())

    def get_paths(self):
        return utils.get_paths_from_experiment(self.experiment_root_path, "lottery", self.epochs)

    @abc.abstractmethod
    def load_x_data(self, paths):
        raise NotImplementedError("Trying to call load_x_data from abstract base class SingleReplicateHandler.")

    @abc.abstractmethod
    def load_y_data(self, paths):
        raise NotImplementedError("Trying to call load_y_data from abstract base class SingleReplicateHandler.")

    @abc.abstractmethod
    def prepare_data_for_plotting(self):
        raise NotImplementedError("Trying to call prepare_data_for_plotting from abstract base class "
                                  "SingleReplicateHandler.")

    @abc.abstractmethod
    def get_plotter(self):
        raise NotImplementedError("Trying to call get_plotter from abstract base class SingleReplicateHandler.")

    # todo add option to explicitely add replicate numbers


class SparsityAccuracyExperimentEvaluator(ExperimentEvaluator):
    def load_x_data(self, paths):
        self.x_data = load_sparsities_for_experiment(paths)

    def load_y_data(self, paths):
        for i, replicate in enumerate(paths.keys()):
            if i == 0:
                accuracies = np.ones((len(paths.keys()), len(paths[replicate]["accuracy"]))) * (-1)
            accuracies[i] = [utils.load_accuracy(acc) for acc in paths[replicate]["accuracy"]]

        self.y_data = accuracies

    def prepare_data_for_plotting(self):
        # nothing to do for x_data
        self.y_data = (list(np.mean(self.y_data, axis=0)), list(np.std(self.y_data, axis=0)))

    def get_plotter(self):
        return plotters.SparsityAccuracyExperimentPlotter(self.num_replicates)


def load_sparsities_for_experiment(paths):
    for i, replicate in enumerate(paths.keys()):
        if i == 0:
            first_replicate = replicate
            sparsities = [utils.load_sparsity(spars) for spars in paths[replicate]["sparsity"]]
        else:
            sparsities_to_be_checked = [utils.load_sparsity(spars) for spars in paths[replicate]["sparsity"]]
            assert_sparsities_equal(sparsities, sparsities_to_be_checked, first_replicate, replicate)
    return sparsities


def assert_sparsities_equal(expected_sparsities, actual_sparsities, key_expected_sparsities, key_actual_sparsities):
    assert actual_sparsities == expected_sparsities, \
        "The sparsities in replicate {} differ from the sparsities in replicate {}, although they should " \
        "be equal. The sparsities in question are {} and {} respectively. Please make sure that you did " \
        "not mix up any experiments.".format(key_expected_sparsities, key_actual_sparsities,
                                             expected_sparsities, actual_sparsities)


class SparsityNeuralPersistenceExperimentEvaluator(ExperimentEvaluator):
    def load_x_data(self, paths):  # loads sparsities
        self.x_data = load_sparsities_for_experiment(paths)

    def load_y_data(self, paths):  # loads neural persistences
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
        self.y_data = neural_persistences

    def prepare_data_for_plotting(self):
        pass

    def get_plotter(self):
        pass


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
    plt.legend()
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
    sparsity_accuracy_plot_experiment("../experiments/lottery_37adeb06fd584c18ebbf48beec5747d3/", 20, show_plot=False, save_plot=True)
    sparsity_neural_persistence_plot_experiment("../experiments/lottery_37adeb06fd584c18ebbf48beec5747d3/", 20, show_plot=False, save_plot=True)
    accuracy_neural_persistence_plot_experiment("../experiments/lottery_37adeb06fd584c18ebbf48beec5747d3/", 20, show_plot=False, save_plot=True)
