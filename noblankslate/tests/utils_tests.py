from collections import OrderedDict
import unittest

import numpy as np
import numpy.testing
import tensorflow as tf
import torch
import torch.nn as nn

import src.utils as utils


# simplified openLTH model todo maybe fix paths later and use real openLTH model.
class TestModel(nn.Module):
    def __init__(self, plan, outputs=10):
        super(TestModel, self).__init__()

        layers = []
        current_size = 4
        for size in plan:
            layers.append(nn.Linear(current_size, size))
            current_size = size

        self.fc_layers = nn.ModuleList(layers)
        self.fc = nn.Linear(current_size, outputs)
        for l in self.fc_layers:
            nn.init.constant_(l.weight, -1)
        nn.init.constant_(self.fc.weight, -1)


class TestPrepareOrderedDictFromModel(unittest.TestCase):
    def test_on_vanilla_ff(self):

        model = TestModel([2], outputs=2)
        expected_keys = ['fc_layers.0.weight', 'fc.weight']
        expected_fc_layers_0_weight = torch.tensor([[-1., -1., -1., -1.], [-1., -1., -1., -1.]])
        expected_fc_weight = torch.tensor([[-1., -1.], [-1., -1.]])

        weights = utils.prepare_ordered_dict_from_model(model)

        self.assertEqual(expected_keys, list(weights.keys()))
        np.testing.assert_array_equal(expected_fc_layers_0_weight, weights['fc_layers.0.weight'])
        np.testing.assert_array_equal(expected_fc_weight, weights['fc.weight'])


class TestLoadMaskedNetwork(unittest.TestCase):
    def test_loading_on_vanilla_ff(self):
        model = TestModel([2], outputs=2)
        torch.save(model, "./resources/test_load_masked_network/model_ep2_it0.pth")
        mask = {'fc_layers.0.weight': torch.tensor([[1., 0., 1., 0.], [0., 1., 1., 0.]]),
                'fc.weight': torch.tensor([[0., 0.], [1., 1.]])}
        torch.save({k: v.int() for k, v in mask.items()}, "./resources/test_load_masked_network/mask.pth")

        expected_keys = ['fc_layers.0.weight', 'fc.weight']
        expected_fc_layers_0_masked_weight = torch.tensor([[-1., 0., -1., 0.], [0., -1., -1., 0.]])
        expected_fc_masked_weight = torch.tensor([[0., 0.], [-1., -1.]])

        weights = utils.load_masked_network("./resources/test_load_masked_network", 2)

        self.assertEqual(expected_keys, list(weights.keys()))
        numpy.testing.assert_array_equal(expected_fc_layers_0_masked_weight, weights["fc_layers.0.weight"])
        numpy.testing.assert_array_equal(expected_fc_masked_weight, weights["fc.weight"])


if __name__ == '__main__':
    unittest.main()
