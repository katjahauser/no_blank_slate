from collections import OrderedDict


def prepare_ordered_dict_from_model(model):
    '''
    tda.PerLayerCalculation requires the weights of a model to be presented as an OrderedDict. Due to changes from
    Tensorflow 1.6 to 2.4, their utility function cannot be used under TF 2.4 out-of-the-box anymore. This function is
    a substitute.

    This function works with the implicit assumption, that the weights are the first item that is returned when calling
    layer.get_weights().

    :param model: A TF model.
    :return: An OrderedDict containing layer_name:numpy_weights tuples.
    '''

    weights = OrderedDict()
    for i, layer in enumerate(model.layers):
        if len(layer.get_weights()) > 0:
            weights.update({layer.name: layer.get_weights()[0]})
    return weights
