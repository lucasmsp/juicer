# -*- coding: utf-8 -*-
import itertools
from gettext import gettext
from textwrap import dedent

from juicer.operation import Operation


# TODO: Por enquanto apenas numeros são suportados. Para liberar para outros fazer via
# cuDF manualmente (gera uma lista de indices e depois split)
class SplitOperation(Operation):
    """
    Randomly splits a Data Frame into two data frames.
    Parameters:
    - List with two weights for the two new data frames.
    - Optional seed in case of deterministic random operation
        ('0' means no seed).

    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        self.weights = float(self.parameters.get('weights', 50)) / 100
        self.seed = self.parameters.get('seed', 'None')
        if type(self.seed) == int:
            self.seed = 0 if self.seed >= 4294967296 or \
                             self.seed < 0 else self.seed
        self.out1 = self.named_outputs.get('split 1',
                                           'split_1_task_{}'.format(self.order))
        self.out2 = self.named_outputs.get('split 2',
                                           'split_2_task_{}'.format(self.order))

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.out2, self.out1])

    def generate_code(self):
        if self.has_code:
            code = """
            import cuml.model_selection as cuml_model_selection
            {out1}, {out2} = cuml_model_selection.train_test_split({input}, train_size = {weights}, shuffle=True)
            """.format(out1=self.out1, out2=self.out2,
                       input=self.named_inputs['input data'],
                       seed=self.seed, weights=self.weights)
            return dedent(code)