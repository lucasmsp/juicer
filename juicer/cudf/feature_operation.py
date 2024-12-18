from textwrap import dedent
from juicer.operation import Operation


import json
try:
    from itertools import zip_longest as zip_longest
except ImportError:
    from itertools import zip_longest as zip_longest


class MinMaxScalerOperation(Operation):
    """
    Transforms features by scaling each feature to a given range.

    This estimator scales and translates each feature individually
    such that it is in the given range on the training set, i.e.
    between zero and one.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attributes'
    MIN_PARAM = 'min'
    MAX_PARAM = 'max'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.ATTRIBUTE_PARAM not in parameters:
            raise ValueError(
                _("Parameters '{}' must be informed for task {}")
                    .format(self.ATTRIBUTE_PARAM, self.__class__))
        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))
        self.model = named_outputs.get(
            'transformation model', 'model_{}'.format(self.order))
        self.attributes = parameters[self.ATTRIBUTE_PARAM]
        self.alias = parameters.get(self.ALIAS_PARAM)
        if self.alias is None:
            self.alias = [col + "_norm" for col in self.attributes]
        else:
            self.alias = self.alias.replace(" ", "").split(",")

        self.min = parameters.get(self.MIN_PARAM, 0)
        self.max = parameters.get(self.MAX_PARAM, 1)


    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=','):
        return sep.join([self.output, self.model])

    def generate_code(self):
        if self.has_code:
            """Generate code."""
            code = """
            
            import cuml.preprocessing as cuml_feature
            X_train = {input}[{att}]

            {model} = cuml_feature.MinMaxScaler(feature_range=({min},{max}))
            {model}.fit(X_train)
            
            values = {model}.transform(X_train)
            values.columns = {alias}

            {output} = cudf.concat([{input}, values],
                ignore_index=False, axis=1)
            """.format(output=self.output, model=self.model,
                       input=self.named_inputs['input data'],
                       att=self.attributes, alias=self.alias,
                       min=self.min, max=self.max)

            return dedent(code)


class MaxAbsScalerOperation(Operation):
    """
    Scale each feature by its maximum absolute value.

    This estimator scales and translates each feature individually
    such that the maximal absolute value of each feature in the training
    set will be 1.0. It does not shift/center the data, and thus does not
     destroy any sparsity.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.ATTRIBUTE_PARAM not in parameters:
            raise ValueError(
                    _("Parameters '{}' must be informed for task {}")
                    .format(self.ATTRIBUTE_PARAM, self.__class__))

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))
        self.model = named_outputs.get(
            'transformation model', 'model_{}'.format(self.order))
        self.attributes = parameters[self.ATTRIBUTE_PARAM]
        self.alias = parameters.get(self.ALIAS_PARAM)
        if self.alias is None:
            self.alias = [col + "_norm" for col in self.attributes]
        else:
            self.alias = self.alias.replace(" ", "").split(",")


    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=','):
        return sep.join([self.output, self.model])

    def generate_code(self):
        """Generate code."""
        if self.has_code:
            code = """
            import cuml.preprocessing as cuml_feature
            X_train = {input}[{att}]

            {model} = cuml_feature.MaxAbsScaler()
            {model}.fit(X_train)
            
            values = {model}.transform(X_train)
            values.columns = {alias}
            
            {output} = cudf.concat([{input}, values],
                ignore_index=False, axis=1)
            """.format(output=self.output, model=self.model,
                       input=self.named_inputs['input data'],
                       att=self.attributes, alias=self.alias)

            return dedent(code)


class StandardScalerOperation(Operation):
    """
    Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing the
    relevant statistics on the samples in the training set. Mean and standard
    deviation are then stored to be used on later data using the transform
    method.

    Standardization of a dataset is a common requirement for many machine
    learning estimators: they might behave badly if the individual feature
    do not more or less look like standard normally distributed data.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attributes'
    WITH_MEAN_PARAM = 'with_mean'
    WITH_STD_PARAM = 'with_std'
    VALUE_PARAMETER = 'value'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(named_inputs) > 0 and any(
            [len(self.named_outputs) > 0, self.contains_results()])
        if self.has_code:
            self.with_mean = parameters.get(
                self.WITH_MEAN_PARAM, False) in ['1', 1, True]
            self.with_std = parameters.get(
                self.WITH_STD_PARAM, True) in ['1', 1, True]
            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))
            self.model = named_outputs.get(
                'transformation model', 'model_{}'.format(self.order))

            if self.ATTRIBUTE_PARAM not in self.parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.ATTRIBUTE_PARAM, self.__class__.__name__))
            self.attributes = parameters[self.ATTRIBUTE_PARAM]
            self.alias = parameters.get(self.ALIAS_PARAM)
            if self.alias is None:
                self.alias = [col + "_norm" for col in self.attributes]
            else:
                self.alias = self.alias.replace(" ", "").split(",")


    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=','):
        return sep.join([self.output, self.model])

    def generate_code(self):
        if self.has_code:
            """Generate code."""
            op = "with_mean={value}" \
                .format(value=self.with_mean)
            op += ", with_std={value}" \
                .format(value=self.with_std)

            code = """
            import cuml.preprocessing as cuml_feature
            X_train = {input}[{att}]

            {model} = cuml_feature.StandardScaler({op})
            {model}.fit(X_train)
            
            values = {model}.transform(X_train)
            values.columns = {alias}

            {output} = cudf.concat([{input}, values],
                ignore_index=False, axis=1)
            """.format(model=self.model, output=self.output,
                       input=self.named_inputs['input data'],
                       att=self.attributes, alias=self.alias, op=op)

            return dedent(code)
