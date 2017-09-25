# -*- coding: utf-8 -*-
import logging

from juicer.runner import configuration

log = logging.getLogger()
log.setLevel(logging.DEBUG)


# noinspection PyClassHasNoInit
class ResultType:
    VISUALIZATION = 'VISUALIZATION'
    MODEL = 'MODEL'


class Operation(object):
    """ Defines an operation in Lemonade """
    __slots__ = ('parameters', 'named_inputs', 'output',
                 'named_outputs', 'multiple_inputs', 'has_code',
                 'expected_output_ports', 'out_degree', 'order',
                 'supports_cache', 'config')

    def __init__(self, parameters, named_inputs, named_outputs):
        self.parameters = parameters
        self.named_inputs = named_inputs
        self.named_outputs = named_outputs
        self.multiple_inputs = False
        self.out_degree = 0

        self.config = configuration.get_config()
        # Assume default as 1, useful for testing.
        self.order = parameters.get('order', 1)

        # Shoud data be cached between job executions?
        # Exception to this rule includes visualization operations.
        self.supports_cache = True

        # Indicate if operation generates code or not. Some operations, e.g.
        self.has_code = len(self.named_inputs) > 0 or len(
            self.named_outputs) > 0

        # How many output ports the operation has
        self.expected_output_ports = 1

        # self.output = 'out_task_{order}'.format(order=parameters['order'])
        # # @!CHECK-ME inspect this part of code.
        # if len(self.named_inputs) > 0:
        #     outputs = self.named_outputs.keys()
        #     self.output = outputs[0] if len(
        #         # self.outputs) > 0 else '{}_tmp_{}'.format(
        #         # self.inputs[0], parameters['task']['order'])
        #         # Used for tests, not correct.
        #         self.outputs) > 0 else '{}_tmp_{}_{}'.format(
        #         self.inputs[0], self.inputs[0],
        #         self.parameters.get('task', {}).get('order', ''))
        #     # Some cases this string to _tmp_ doesn't work in the spark code generation
        #     #  parameters['task']['order']
        # elif len(self.outputs) > 0:
        #     self.output = self.outputs[0]
        # else:
        #     self.output = "NO_OUTPUT_WITHOUT_CONNECTIONS"

        # Subclasses should override this
        self.output = self.named_outputs.get(
            'output data', 'out_task_{}'.format(self.order))

    def generate_code(self):
        raise NotImplementedError(
            _("Method generate_code should be implemented "\
              "in {} subclass").format(self.__class__))

    def get_generated_results(self):
        """
         Returns results generated by a task executing an operation.
         Results can be models and visualizations (for while).
        """
        return []

    @property
    def get_inputs_names(self):
        return ', '.join(self.named_inputs.values())

    def get_output_names(self, sep=", "):
        return sep.join(self.named_outputs.values())

    def get_data_out_names(self, sep=','):
        return self.get_output_names(sep)


# noinspection PyAbstractClass
class ReportOperation(Operation):
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)


# noinspection PyAbstractClass
class NoOp(Operation):
    """ Null operation """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = False
