import abc
import os
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt


class PlotterBaseClass(metaclass=abc.ABCMeta):
    """"""
    title = abc.abstractproperty()
    x_label = abc.abstractproperty()
    y_label = abc.abstractproperty()
    axis = abc.abstractproperty()
    figure = abc.abstractproperty()
    save_file_name = abc.abstractproperty()

    # title and labels as abstract arguments (oder so)
    # ggf aufsplitten in prepare data for plotting und make plot? Eigentlich besser eine prepare Funktion/Klasse
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'plot_data') and
                callable(subclass.make_plot) and
                hasattr(subclass, 'title') and
                hasattr(subclass, 'x_label') and
                hasattr(subclass, 'y_label') and
                hasattr(subclass, 'axis') and
                hasattr(subclass, 'figure') and
                hasattr(subclass, 'save_file_name'))

    def make_plot(self, x_data, y_data):
        self.axis = self.setup_figure_and_axis()
        self.plot_data(self.axis, x_data, y_data)
        self.set_title_and_axis_labels(self.axis)

    def setup_figure_and_axis(self):
        # Strictly speaking this function does more than one thing, but it encapsulates one call, so that should be
        # fine.
        # It returns an axis to enforce being called before any functions that work on an axis (e.g., plot_data and
        # set_title_and_axis_labels).
        # The figure is necessary to avoid introducing an implicit order of calls (save before show) by using
        # Figure.savefig instead of matplotlib.pyplot.savefig. See the Notes here
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html for a more detailed explanation.
        self.figure, axis = plt.subplots(1, 1)
        return axis

    @abc.abstractmethod
    def plot_data(self, axis, x_values, y_values):
        # plot_data takes an axis object to enforce calling setup_axis before calling plot_data
        raise NotImplementedError("Trying to call plot_data of abstract class PlotterBaseClass.")

    def set_title_and_axis_labels(self, axis):
        # set_title_and_axis_labels takes an axis object to enforce calling setup_axis before calling
        # set_title_and_axis_labels
        axis.set_title(self.title)
        axis.set_xlabel(self.x_label)
        axis.set_ylabel(self.y_label)

    def save_plot(self, experiment_root_path):
        path_handler = ReplicatePathHandler()
        save_dir = path_handler.prepare_plot_dir(experiment_root_path)
        path_handler.make_dir_if_does_not_exist(save_dir)
        save_path = path_handler.create_save_path(save_dir, self.save_file_name)

        self.figure.savefig(save_path)

    def show_plot(self):
        if self.axis is None:
            raise(TypeError, "self.axis is None -- you must run make_plot or setup_plot before you can show a plot.")
        else:
            plt.show()

    def __del__(self):
        # When I ran a test suit for an object that contains a derivative of PlotterBaseClass I got more open plots
        # than expected when running the full suit, but not when running the tests individually. This destructor fixes
        # the issue.
        plt.close(self.figure)


# todo add functionality to deal with several replicates (i.e. take custom names)
class ReplicatePathHandler:
    @staticmethod
    def prepare_plot_dir(experiment_root):
        return str(Path(experiment_root).parent) + "/plots/"

    @staticmethod
    def make_dir_if_does_not_exist(plot_dir):
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

    @staticmethod
    def create_save_path(path, file_name):
        if path[-1] != "/":
            path = path + "/"
        return path + file_name


class SparsityAccuracyReplicatePlotter(PlotterBaseClass):
    """"""
    title = "Sparsity-Accuracy"
    x_label = "Sparsity"
    y_label = "Accuracy"
    axis = None
    figure = None
    save_file_name = "sparsity_accuracy_replicate_plot.png"

    def plot_data(self, axis, sparsities, accuracies):
        axis.plot(sparsities, accuracies)
        axis.invert_xaxis()