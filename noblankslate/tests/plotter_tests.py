import os
import unittest

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

import src.plotters as plotters


show_no_plots_for_automated_tests = True


class ConcretePlotter(plotters.PlotterBaseClass):
    # This is a mock class used to test the non-abstract methods in plotters.PlotterBaseClass
    title = "ConcretePlotter Title"
    x_label = "ConcretePlotter x-label"
    y_label = "ConcretePlotter y-label"
    save_file_name = "ConcretePlotter_plot.png"

    def plot_data(self, axis, x_values, y_values):
        axis.plot(x_values, y_values)


class TestPlotterBaseClass(unittest.TestCase):
    def test_is_base_plotter_subclass(self):
        plotter = ConcretePlotter()

        self.assertTrue(isinstance(plotter, plotters.PlotterBaseClass))
        self.assertTrue(issubclass(plotters.SparsityAccuracyReplicatePlotter, plotters.PlotterBaseClass))

    def test_class_variables_set_correctly(self):
        plotter = ConcretePlotter()

        self.assertEqual("ConcretePlotter Title", plotter.title)
        self.assertEqual("ConcretePlotter x-label", plotter.x_label)
        self.assertEqual("ConcretePlotter y-label", plotter.y_label)
        self.assertEqual(None, plotter.axis)
        self.assertEqual(None, plotter.figure)
        self.assertEqual("ConcretePlotter_plot.png", plotter.save_file_name)

    def test_setup_figure_and_axis(self):
        plotter = ConcretePlotter()

        plotter.axis = plotter.setup_figure_and_axis()

        self.assertTrue(issubclass(type(plotter.axis), matplotlib.axes.SubplotBase))
        self.assertTrue(issubclass(type(plotter.figure), matplotlib.figure.Figure))

    def test_set_title_and_axis_labels(self):
        plotter = ConcretePlotter()
        plotter.axis = plotter.setup_figure_and_axis()

        plotter.set_title_and_axis_labels(plotter.axis)
        self.assertEqual("ConcretePlotter Title", plotter.axis.get_title())
        self.assertEqual("ConcretePlotter x-label", plotter.axis.get_xlabel())
        self.assertEqual("ConcretePlotter y-label", plotter.axis.get_ylabel())

    def test_save_plot(self):
        test_path = "resources/test_plots/plots/ConcretePlotter_plot.png"
        if os.path.exists(test_path):
            os.remove(test_path)
        assert not os.path.exists(test_path)
        plotter = ConcretePlotter()
        x_data = np.arange(3)
        y_data = np.ones(3)
        plotter.make_plot(x_data, y_data)

        plotter.save_plot("resources/test_plots/empty_dummy_root")

        self.assertTrue(os.path.exists(test_path))

    def test_show_plot_shows_when_axis_not_none(self):
        if not show_no_plots_for_automated_tests:
            plotter = ConcretePlotter()
            x_data = np.arange(3)
            y_data = np.ones(3)
            plotter.make_plot(x_data, y_data)

            plotter.show_plot()
        else:
            print("Ignoring TestSparsityAccuracyReplicatePlotter.test_show_plot_shows_when_axis_not_none to allow "
                  "automated testing.")

    def test_show_plot_raises_when_axis_none(self):
        plotter = ConcretePlotter()

        with self.assertRaises(TypeError):
            plotter.show_plot()

    def test_deletion(self):
        plotter = ConcretePlotter()
        plotter.setup_figure_and_axis()
        # only one figure exists -- this is relevant, if the test suit runs several tests in parallel
        assert(len(plt.get_fignums()) == 1)

        plotter.__del__()

        self.assertFalse(plt.get_fignums())  # an empty list is cast to False by convention


class TestReplicatePathHandler(unittest.TestCase):
    def test_prepare_plot_dir(self):
        test_path = "./experiment_root/"
        test_path_wo_trailing_slash = "./experiment_root"
        path_handler = plotters.ReplicatePathHandler()

        plot_dir = path_handler.prepare_plot_dir(test_path)
        plot_dir_from_path_wo_trailing_slash = path_handler.prepare_plot_dir(test_path_wo_trailing_slash)

        self.assertEqual("./plots/", plot_dir)
        self.assertEqual("./plots/", plot_dir_from_path_wo_trailing_slash)

    def test_make_dir_if_does_not_exist(self):
        if os.path.isdir("./resources/test_plots/make_dir_if_does_not_exist"):
            os.removedirs("./resources/test_plots/make_dir_if_does_not_exist")
        assert not os.path.exists("./resources/test_plots/make_dir_if_does_not_exist")
        path_handler = plotters.ReplicatePathHandler()

        path_handler.make_dir_if_does_not_exist("./resources/test_plots/make_dir_if_does_not_exist")

        self.assertTrue(os.path.isdir("./resources/test_plots/make_dir_if_does_not_exist"))

    def test_leave_dir_alone_if_exists(self):
        assert os.path.exists("./resources/test_plots/plots/")
        assert os.path.exists("./resources/test_plots/plots/accuracy_neural_persistence.png")
        path_handler = plotters.ReplicatePathHandler()

        path_handler.make_dir_if_does_not_exist("./resources/test_plots/plots/")

        self.assertTrue(os.path.exists("./resources/test_plots/plots/"))
        self.assertTrue(os.path.exists("./resources/test_plots/plots/accuracy_neural_persistence.png"))

    def test_create_save_path(self):
        path_handler = plotters.ReplicatePathHandler()
        dir_with_trailing_slash = "./test/plots/"
        dir_wo_trailing_slash = "./test/plots"

        path_with_trailing_slash = path_handler.create_save_path(dir_with_trailing_slash, "my_plot.png")
        path_wo_trailing_slash = path_handler.create_save_path(dir_wo_trailing_slash, "my_plot.png")

        self.assertEqual("./test/plots/my_plot.png", path_with_trailing_slash)
        self.assertEqual("./test/plots/my_plot.png", path_wo_trailing_slash)


class TestSparsityAccuracyReplicatePlotter(unittest.TestCase):
    def test_is_base_plotter_subclass(self):
        plotter = plotters.SparsityAccuracyReplicatePlotter()

        self.assertTrue(isinstance(plotter, plotters.PlotterBaseClass))
        self.assertTrue(issubclass(plotters.SparsityAccuracyReplicatePlotter, plotters.PlotterBaseClass))

    def test_class_variables_set_correctly(self):
        plotter = plotters.SparsityAccuracyReplicatePlotter()

        self.assertEqual("Sparsity-Accuracy", plotter.title)
        self.assertEqual("Sparsity", plotter.x_label)
        self.assertEqual("Accuracy", plotter.y_label)
        self.assertEqual("sparsity_accuracy_replicate_plot.png", plotter.save_file_name)

    def test_axis_has_data(self):
        # Following the reasoning in https://stackoverflow.com/a/27950953 I'm checking the desired output of the
        # function that does the plotting, but not the plot itself
        plotter = plotters.SparsityAccuracyReplicatePlotter()
        x_data = np.arange(3)
        y_data = np.ones(3)
        plotter.axis = plotter.setup_figure_and_axis()
        assert not plotter.axis.has_data()

        plotter.plot_data(plotter.axis, x_data, y_data)

        self.assertTrue(plotter.axis.has_data())

    def test_inversion_of_x_axis(self):
        plotter = plotters.SparsityAccuracyReplicatePlotter()
        x_data = np.arange(3)
        y_data = np.ones(3)
        plotter.axis = plotter.setup_figure_and_axis()

        plotter.plot_data(plotter.axis, x_data, y_data)

        self.assertTrue(plotter.axis.xaxis_inverted())

    def test_show_plot(self):
        # sanity check
        if not show_no_plots_for_automated_tests:
            plotter = plotters.SparsityAccuracyReplicatePlotter()
            x_data = np.arange(3)
            y_data = np.ones(3)
            plotter.make_plot(x_data, y_data)

            plotter.show_plot()


class TestSparsityNeuralPersistenceReplicatePlotter(unittest.TestCase):
    def test_is_base_plotter_subclass(self):
        plotter = plotters.SparsityNeuralPersistenceReplicatePlotter()

        self.assertTrue(isinstance(plotter, plotters.PlotterBaseClass))
        self.assertTrue(issubclass(plotters.SparsityNeuralPersistenceReplicatePlotter, plotters.PlotterBaseClass))

    def test_class_variables_set_correctly(self):
        plotter = plotters.SparsityNeuralPersistenceReplicatePlotter()

        self.assertEqual("Sparsity-Neural Persistence", plotter.title)
        self.assertEqual("Sparsity", plotter.x_label)
        self.assertEqual("Neural Persistence", plotter.y_label)
        self.assertEqual("sparsity_neural_persistence_replicate_plot.png", plotter.save_file_name)

    def test_axis_has_data(self):
        # Following the reasoning in https://stackoverflow.com/a/27950953 I'm checking the desired output of the
        # function that does the plotting, but not the plot itself
        plotter = plotters.SparsityNeuralPersistenceReplicatePlotter()
        x_data = np.arange(3)
        y_data = {"test1": np.ones(3), "test2": np.ones(3)*2}
        plotter.axis = plotter.setup_figure_and_axis()
        assert not plotter.axis.has_data()

        plotter.plot_data(plotter.axis, x_data, y_data)

        self.assertTrue(plotter.axis.has_data())

    def test_inversion_of_x_axis(self):
        plotter = plotters.SparsityNeuralPersistenceReplicatePlotter()
        x_data = np.arange(3)
        y_data = {"test1": np.ones(3), "test2": np.ones(3)*2}
        plotter.axis = plotter.setup_figure_and_axis()

        plotter.plot_data(plotter.axis, x_data, y_data)

        self.assertTrue(plotter.axis.xaxis_inverted())

    def test_has_legend(self):
        plotter = plotters.SparsityNeuralPersistenceReplicatePlotter()
        x_data = np.arange(3)
        y_data = {"test1": np.ones(3), "test2": np.ones(3)*2}
        plotter.axis = plotter.setup_figure_and_axis()

        plotter.plot_data(plotter.axis, x_data, y_data)

        self.assertIsNotNone(plotter.axis.get_legend())

    def test_show_plot(self):
        # sanity check
        if not show_no_plots_for_automated_tests:
            plotter = plotters.SparsityNeuralPersistenceReplicatePlotter()
            x_data = np.arange(3)
            y_data = {"test1": np.ones(3), "test2": np.ones(3)*2}
            plotter.make_plot(x_data, y_data)

            plotter.show_plot()


class TestAccuracyNeuralPersistenceReplicatePlotter(unittest.TestCase):
    def test_is_base_plotter_subclass(self):
        plotter = plotters.AccuracyNeuralPersistenceReplicatePlotter()

        self.assertTrue(isinstance(plotter, plotters.PlotterBaseClass))
        self.assertTrue(
            issubclass(plotters.AccuracyNeuralPersistenceReplicatePlotter, plotters.PlotterBaseClass))

    def test_class_variables_set_correctly(self):
        plotter = plotters.AccuracyNeuralPersistenceReplicatePlotter()

        self.assertEqual("Accuracy-Neural Persistence", plotter.title)
        self.assertEqual("Accuracy", plotter.x_label)
        self.assertEqual("Neural Persistence", plotter.y_label)
        self.assertEqual("accuracy_neural_persistence_replicate_plot.png", plotter.save_file_name)

    def test_axis_has_data(self):
        # Following the reasoning in https://stackoverflow.com/a/27950953 I'm checking the desired output of the
        # function that does the plotting, but not the plot itself
        plotter = plotters.AccuracyNeuralPersistenceReplicatePlotter()
        x_data = np.arange(3)
        y_data = {"test1": np.ones(3), "test2": np.ones(3)*2}
        plotter.axis = plotter.setup_figure_and_axis()
        assert not plotter.axis.has_data()

        plotter.plot_data(plotter.axis, x_data, y_data)

        self.assertTrue(plotter.axis.has_data())

    def test_has_legend(self):
        plotter = plotters.AccuracyNeuralPersistenceReplicatePlotter()
        x_data = np.arange(3)
        y_data = {"test1": np.ones(3), "test2": np.ones(3)*2}
        plotter.axis = plotter.setup_figure_and_axis()

        plotter.plot_data(plotter.axis, x_data, y_data)

        self.assertIsNotNone(plotter.axis.get_legend())

    def test_show_plot(self):
        # sanity check
        if not show_no_plots_for_automated_tests:
            plotter = plotters.AccuracyNeuralPersistenceReplicatePlotter()
            x_data = np.arange(3)
            y_data = {"test1": np.ones(3), "test2": np.ones(3) * 2}
            plotter.make_plot(x_data, y_data)

            plotter.show_plot()


class TestExperimentPlotterBaseClass(unittest.TestCase):
    def test_inheritance(self):
        plotter = ConcreteExperimentPlotter(1)

        self.assertTrue(isinstance(plotter, plotters.PlotterBaseClass))
        self.assertTrue(isinstance(plotter, plotters.ExperimentPlotterBaseClass))
        self.assertTrue(issubclass(plotters.ExperimentPlotterBaseClass, plotters.PlotterBaseClass))
        self.assertTrue(issubclass(ConcreteExperimentPlotter, plotters.ExperimentPlotterBaseClass))
        self.assertTrue(issubclass(ConcreteExperimentPlotter, plotters.PlotterBaseClass))

    def test_optional_set_title_upon_initialization(self):
        expected_title_with_replicates = "ConcreteExperimentPlotter over 2 replicates"
        expected_title_wo_replicates = "ConcreteExperimentPlotter"

        plotter_with_replicates = ConcreteExperimentPlotter(2)
        plotter_wo_replicates = ConcreteExperimentPlotter()

        self.assertEqual(expected_title_with_replicates, plotter_with_replicates.title)
        self.assertEqual(expected_title_wo_replicates, plotter_wo_replicates.title)

    def test_generate_title_if_valid_num_replicates(self):
        plotter = ConcreteExperimentPlotter()
        expected_title = "ConcreteExperimentPlotter over 2 replicates"

        actual_title = plotter.generate_title_if_valid_num_replicates(2)

        self.assertEqual(expected_title, actual_title)

    def test_generate_title_if_valid_num_replicates_on_1_replicate(self):
        plotter = ConcreteExperimentPlotter(2)
        expected_title = "ConcreteExperimentPlotter over 1 replicate"
        plotter.title = "ConcreteExperimentPlotter"

        actual_title = plotter.generate_title_if_valid_num_replicates(1)

        self.assertEqual(expected_title, actual_title)

    def test_generate_title_if_valid_num_replicates_raises_on_invalid_input(self):
        equals_0 = 0
        smaller_0 = -5
        smaller_0_and_not_int = -9.8
        not_int = 4.5
        valid_num_replicates = 2
        plotter = ConcreteExperimentPlotter(valid_num_replicates)

        with self.assertRaises(ValueError):
            plotter.generate_title_if_valid_num_replicates(equals_0)
        with self.assertRaises(ValueError):
            plotter.generate_title_if_valid_num_replicates(smaller_0)
        with self.assertRaises(ValueError):
            plotter.generate_title_if_valid_num_replicates(smaller_0_and_not_int)
        with self.assertRaises(ValueError):
            plotter.generate_title_if_valid_num_replicates(not_int)


class ConcreteExperimentPlotter(plotters.ExperimentPlotterBaseClass):
    title = "ConcreteExperimentPlotter"
    x_label = ""
    y_label = ""
    save_file_name = ""

    def plot_data(self, axis, x_values, y_values):
        pass


class TestSparsityAccuracyExperimentPlotter(unittest.TestCase):
    def test_inheritance(self):
        plotter = plotters.SparsityAccuracyExperimentPlotter()

        self.assertTrue(isinstance(plotter, plotters.PlotterBaseClass))
        self.assertTrue(
            issubclass(plotters.SparsityAccuracyExperimentPlotter, plotters.ExperimentPlotterBaseClass))
        self.assertTrue(
            issubclass(plotters.SparsityAccuracyExperimentPlotter, plotters.PlotterBaseClass))

    def test_class_variables_set_correctly_wo_num_replicate(self):
        plotter = plotters.SparsityAccuracyExperimentPlotter()

        self.assertEqual("Sparsity-Accuracy Experiment", plotter.title)
        self.assertEqual("Sparsity", plotter.x_label)
        self.assertEqual("Accuracy", plotter.y_label)
        self.assertEqual("sparsity_accuracy_experiment_plot.png", plotter.save_file_name)

    def test_title_set_correctly_with_num_replicate(self):
        plotter = plotters.SparsityAccuracyExperimentPlotter(2)

        self.assertEqual("Sparsity-Accuracy Experiment over 2 replicates", plotter.title)

    def test_axis_has_data(self):
        # Following the reasoning in https://stackoverflow.com/a/27950953 I'm checking the desired output of the
        # function that does the plotting, but not the plot itself
        plotter = plotters.SparsityAccuracyExperimentPlotter()
        x_data = list(np.arange(3))
        y_data = (list(np.ones(3)), list(np.arange(3)*0.5))
        plotter.axis = plotter.setup_figure_and_axis()
        assert not plotter.axis.has_data()

        plotter.plot_data(plotter.axis, x_data, y_data)

        self.assertTrue(plotter.axis.has_data())

    def test_inversion_of_x_axis(self):
        plotter = plotters.SparsityAccuracyExperimentPlotter()
        x_data = list(np.arange(3))
        y_data = (list(np.ones(3)), list(np.arange(3)*0.5))
        axis = plotter.setup_figure_and_axis()

        plotter.plot_data(axis, x_data, y_data)

        self.assertTrue(axis.xaxis_inverted())

    def test_show_plot(self):
        # sanity check
        if not show_no_plots_for_automated_tests:
            plotter = plotters.SparsityAccuracyExperimentPlotter(3)
            x_data = list(np.arange(3))
            y_data = (list(np.ones(3)), list(np.arange(3)*0.5))
            plotter.make_plot(x_data, y_data)

            plotter.show_plot()


if __name__ == '__main__':
    unittest.main()
