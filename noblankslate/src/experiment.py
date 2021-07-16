import abc
import collections

import numpy as np

import src.plotters as plotters
import src.utils as utils

# todo add plot of NP of mask vs NP of masked weights
# todo add replicate number/name to replicate plotting function save paths


class ReplicateEvaluator(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_x_data') and
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

    def get_paths(self):
        return utils.get_paths_from_replicate(self.experiment_root_path, "lottery", self.epochs)

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
    def load_x_data(self, paths):
        self.x_data = utils.load_sparsities_of_replicate(paths)

    def load_y_data(self, paths):
        self.y_data = utils.load_accuracies_of_replicate(paths)

    def prepare_data_for_plotting(self):
        pass  # nothing to do for this type of plot

    def get_plotter(self):
        return plotters.SparsityAccuracyReplicatePlotter()


class SparsityNeuralPersistenceOnSingleReplicateEvaluator(ReplicateEvaluator):
    def load_x_data(self, paths):
        self.x_data = utils.load_sparsities_of_replicate(paths)

    def load_y_data(self, paths):
        self.y_data = utils.load_neural_persistences_of_replicate(paths)

    def prepare_data_for_plotting(self):
        # nothing to do for self.x_data
        self.y_data = utils.prepare_neural_persistence_for_plotting(self.y_data)

    def get_plotter(self):
        return plotters.SparsityNeuralPersistenceReplicatePlotter()


class AccuracyNeuralPersistenceOnSingleReplicateEvaluator(ReplicateEvaluator):
    def load_x_data(self, paths):
        self.x_data = utils.load_accuracies_of_replicate(paths)

    def load_y_data(self, paths):
        self.y_data = utils.load_neural_persistences_of_replicate(paths)

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

    # todo add option to explicitely turn showing replicate numbers on/off


class SparsityAccuracyExperimentEvaluator(ExperimentEvaluator):
    def load_x_data(self, paths):
        self.x_data = utils.load_sparsities_of_experiment(paths)

    def load_y_data(self, paths):
        self.y_data = utils.load_accuracies_of_experiment(paths)

    def prepare_data_for_plotting(self):
        # nothing to do for x_data
        self.y_data = (list(np.mean(self.y_data, axis=0)), list(np.std(self.y_data, axis=0)))

    def get_plotter(self):
        return plotters.SparsityAccuracyExperimentPlotter(self.num_replicates)


class NeuralPersistenceExperimentEvaluator(ExperimentEvaluator):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_x_data') and
                callable(subclass.load_x_data) and
                hasattr(subclass, 'prepare_x_data_for_plotting') and
                callable(subclass.prepare_x_data_for_plotting) and
                hasattr(subclass, 'prepare_neural_persistences_for_plotting') and
                callable(subclass.prepare_neural_persistences_for_plotting) and
                hasattr(subclass, 'match_layer_names') and
                callable(subclass.match_layer_names) and
                hasattr(subclass, 'get_plotter') and
                callable(subclass.get_plotter))

    def __init__(self, experiment_root_path, eps):
        super().__init__(experiment_root_path, eps)

    @abc.abstractmethod
    def load_x_data(self, paths):
        raise NotImplementedError("Trying to call load_x_data from abstract base class "
                                  "NeuralPersistenceExperimentEvaluator.")

    def load_y_data(self, paths):
        self.y_data = utils.load_neural_persistences_of_experiment(paths)

    def prepare_data_for_plotting(self):
        layer_names = self.get_layer_names()
        self.reformat_neural_persistences()
        self.prepare_x_data_for_plotting()
        self.prepare_neural_persistences_for_plotting()
        self.match_layer_names(layer_names)

    def get_layer_names(self):
        if type(self.y_data[0][0]) is not collections.defaultdict:
            raise TypeError("y_data has the wrong type ({}). You probably reformated the data before getting the layer "
                            "names.".format(type(self.y_data[0][0])))
        return self.y_data[0][0].keys()

    def reformat_neural_persistences(self):
        # transform neural persistences (y_data) from
        # [[replicate1.sparsity_level0, replicate1.sparsity_level1, ...],
        #  [replicate2.sparsity_level0, replicate2.sparsity_level1, ...],
        #  ...]
        # (each replicateX.sparsity_levelY is a neural persistence dict from tda.PerLayerCalculation)
        # to
        # np.array(shape=(layers, sparsity_levels, replicates))
        reformated_np = self.create_tensor_for_reformating()
        for replicate in range(len(self.y_data)):
            for sparsity_level in range(len(self.y_data[0])):
                # implicitly guarantees that all layer names are the same
                for layer_number, layer_name in enumerate(self.y_data[0][0].keys()):
                    reformated_np[layer_number][sparsity_level][replicate] = \
                        self.get_normalized_neural_persistence(self.y_data[replicate][sparsity_level][layer_name])
        self.y_data = reformated_np

    def create_tensor_for_reformating(self):
        # number of keys (= number of layers + global) for model in first replicate at first sparsity level
        num_layers = len(self.y_data[0][0].keys())
        num_sparsity_levels = len(self.y_data[0])
        return np.zeros((num_layers, num_sparsity_levels, self.num_replicates))

    @staticmethod
    def get_normalized_neural_persistence(layer_neural_persistence_dict):
        for (key, value) in layer_neural_persistence_dict.items():
            if "normalized" in key:
                return value

    @abc.abstractmethod
    def prepare_x_data_for_plotting(self):
        raise NotImplementedError("Trying to call prepare_x_data from abstract base class "
                                  "NeuralPersistenceExperimentEvaluator.")

    @abc.abstractmethod
    def prepare_neural_persistences_for_plotting(self):
        # takes an np.array(shape=(layers, sparsity_levels, replicates)) as an argument.
        # The layer names are extracted by calling .keys(), i.e., the layer names are presented in insertion order,
        # therefore you should not change the ordering of the array along the first axis.
        raise NotImplementedError("Trying to call prepare_neural_persistences from abstract base class "
                                  "NeuralPersistenceExperimentEvaluator.")

    @abc.abstractmethod
    def match_layer_names(self, layer_names):
        raise NotImplementedError("Trying to call match_layer_names from abstract base class "
                                  "NeuralPersistenceExperimentEvaluator.")

    @abc.abstractmethod
    def get_plotter(self):
        raise NotImplementedError("Trying to call get_plotter from abstract base class "
                                  "NeuralPersistenceExperimentEvaluator.")


class SparsityNeuralPersistenceExperimentEvaluator(NeuralPersistenceExperimentEvaluator):
    def load_x_data(self, paths):  # loads sparsities
        self.x_data = utils.load_sparsities_of_experiment(paths)

    def prepare_x_data_for_plotting(self):
        # nothing to do for sparsities
        pass

    def prepare_neural_persistences_for_plotting(self):
        self.y_data = (self.compute_means(), self.compute_std_deviations())

    def compute_means(self):
        return np.mean(self.y_data, axis=2)

    def compute_std_deviations(self):
        return np.std(self.y_data, axis=2)

    def match_layer_names(self, layer_names):
        means = self.y_data[0]
        std_devs = self.y_data[1]
        matched_means = self.match_layer_names_to_statistic(layer_names, means)
        matched_std_devs = self.match_layer_names_to_statistic(layer_names, std_devs)
        self.y_data = (matched_means, matched_std_devs)

    @staticmethod
    def match_layer_names_to_statistic(layer_names, statistic):
        mapped_names = {}
        for i, name in enumerate(layer_names):
            mapped_names.update({name: statistic[i, :]})
        return mapped_names

    def get_plotter(self):
        return plotters.SparsityNeuralPersistenceExperimentPlotter(self.num_replicates)


class AccuracyNeuralPersistenceExperimentEvaluator(NeuralPersistenceExperimentEvaluator):
    def load_x_data(self, paths):
        self.x_data = utils.load_accuracies_of_experiment(paths)

    def prepare_x_data_for_plotting(self):
        pass  # nothing to do for x_data (accuracies)

    def prepare_neural_persistences_for_plotting(self):
        pass  # nothing to do for neural persistences

    def match_layer_names(self, layer_names):
        matched_neural_persistences = {}
        for i, key in enumerate(layer_names):
            matched_neural_persistences.update({key: self.y_data[i, :, :]})
        self.y_data = matched_neural_persistences

    def get_plotter(self):
        return plotters.AccuracyNeuralPersistenceExperimentPlotter(self.num_replicates)


if __name__ == "__main__":

    SparsityAccuracyExperimentEvaluator("../experiments/lottery_37adeb06fd584c18ebbf48beec5747d3/", 20)\
        .evaluate(show_plot=False, save_plot=True)
    SparsityNeuralPersistenceExperimentEvaluator("../experiments/lottery_37adeb06fd584c18ebbf48beec5747d3/", 20)\
        .evaluate(show_plot=False, save_plot=True)
    AccuracyNeuralPersistenceExperimentEvaluator("../experiments/lottery_37adeb06fd584c18ebbf48beec5747d3/", 20)\
        .evaluate(show_plot=False, save_plot=True)
