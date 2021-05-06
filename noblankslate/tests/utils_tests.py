from collections import OrderedDict
import unittest

import numpy as np
import numpy.testing as np_testing
import tensorflow as tf

import src.utils as utils


class TestPrepareOrderedDictFromModel(unittest.TestCase):
    def test_on_vanilla_ff(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, kernel_initializer=tf.keras.initializers.constant(1.)),
            tf.keras.layers.Dense(2, kernel_initializer=tf.keras.initializers.constant(1.))
        ])

        model(tf.ones((2, 2)))

        true_weights = OrderedDict({"dense": np.ones((2, 2)), "dense_1": np.ones((2, 2))})

        weights = utils.prepare_ordered_dict_from_model(model)

        self.assertEqual(true_weights.keys(), weights.keys())
        for k in true_weights.keys():
            self.assertEqual(true_weights[k].tolist(), weights[k].tolist(), "Assertion caused by layer '{}'.".format(k))


if __name__ == '__main__':
    unittest.main()
