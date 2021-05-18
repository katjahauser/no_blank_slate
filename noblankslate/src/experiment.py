import src.utils as utils

# todo merge last commit with next commit in git


# plots:
    # * pruning rate-accuracy as sanity check
    # * pruning rate-NP (variant 1: NP global, variant 2: NP layerwise; potentially solve plotting them as 2 or as 1
    # plot as flag (for networks with more layers; option: add layer grouping?)
    # * accuracy-NP as a sanity check in a way (do both global and layer-wise. potentially put this into different
    # function.)
    # * NP of mask vs NP of masked weights

# todo implement get_sparsity and add sparsity file to get_file_paths

def sparsity_accuracy(experiment_root_path, eps):
    """"""
    paths = utils.get_file_paths(experiment_root_path, "lottery", eps)

    print(paths)
    return 0


def neural_persistence_sparsity(experiment_root_path):
    """"""
    # load data

    # make plots (both as tikz and matplotlib, optional flag for matplotlib plots)
    # come up with clever save paths :P (probably: experiments/plots_experimentname/plot_thingy)

    return 0
# assumption: the open LTH side of the experiment is already run
# works with "lottery" experiment type for now
# answers base question: is there any difference wrt NP in pruned networks depending on the level of sparsity?
# plots: pruning rate vs NP, pruning rate vs accuracy, NP vs accuracy (is there any correlation?)


def accuracy_neural_persistence(experiment_root_path):
    """"""
    pass
