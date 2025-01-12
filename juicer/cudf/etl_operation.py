# -*- coding: utf-8 -*-
import itertools
from gettext import gettext
from textwrap import dedent

from juicer.operation import Operation
from juicer.cudf.expression import Expression


class FilterOperation(Operation):
    """
    Filters rows using the given condition.
    Parameters:
        - The expression (==, <, >)
    """
    FILTER_PARAM = 'filter'
    ADVANCED_FILTER_PARAM = 'expression'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.FILTER_PARAM not in parameters and self.ADVANCED_FILTER_PARAM \
                not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}".format(
                    self.FILTER_PARAM, self.__class__)))

        self.advanced_filter = parameters.get(self.ADVANCED_FILTER_PARAM) or []
        self.filter = parameters.get(self.FILTER_PARAM) or []

        self.has_code = len(named_inputs) > 0 and any(
            [len(self.named_outputs) > 0, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))

    def generate_code(self):
        if self.has_code:
            input_data = self.named_inputs['input data']
            params = {'input': input_data}

            filters = [
                "{0} {1} {2}".format(f['attribute'], f['f'],
                                       f.get('value', f.get('alias')))
                for f in self.filter]

            code = """
            {out} = {input}""".format(out=self.output,
                                      input=self.named_inputs['input data'])

            expressions = []
            for i, expr in enumerate(self.advanced_filter):
                expression = Expression(expr['tree'], params)
                expressions.append(expression.parsed_expression)

            if len(expressions) > 0:
                for e in expressions:
                    code += """
            {out} = {out}[{out}.apply({expr}, axis=1)]"""\
                    .replace(
                        "{out} = {out}[{out}.apply(", 
                        "{out} = {out}.loc[")\
                    .replace(
                        ", axis=1)", 
                        "")\
                    .format(out=self.output, expr=e)\
                    .replace("lambda row:", "")\
                    .replace("row[", self.output+"[")

            indentation = " and "
            if len(filters) > 0:
                code += """
            {out} = {out}.query('{f}')""".format(out=self.output,
                                                 f=indentation.join(filters))

            return dedent(code)

            
class JoinOperation(Operation):
    """
    Joins with another DataFrame, using the given join expression.
    The expression must be defined as a string parameter.
    """
    KEEP_RIGHT_KEYS_PARAM = 'keep_right_keys'
    MATCH_CASE_PARAM = 'match_case'
    JOIN_TYPE_PARAM = 'join_type'
    LEFT_ATTRIBUTES_PARAM = 'left_attributes'
    RIGHT_ATTRIBUTES_PARAM = 'right_attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.not_keep_right_keys = not \
            parameters.get(self.KEEP_RIGHT_KEYS_PARAM, False) in (1, '1', True)
        self.match_case = parameters.get(self.MATCH_CASE_PARAM, False) in (
            1, '1', True)

        self.join_type = parameters.get(self.JOIN_TYPE_PARAM, 'inner')

        # outer should not be allowed?
        self.join_type = self.join_type.replace("_outer", "")

        if not all([self.LEFT_ATTRIBUTES_PARAM in parameters,
                    self.RIGHT_ATTRIBUTES_PARAM in parameters]):
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}").format
                (self.LEFT_ATTRIBUTES_PARAM,
                 self.RIGHT_ATTRIBUTES_PARAM,
                 self.__class__))

        self.has_code = len(self.named_inputs) == 2 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        self.left_attributes = parameters.get(self.LEFT_ATTRIBUTES_PARAM)
        self.right_attributes = parameters.get(self.RIGHT_ATTRIBUTES_PARAM)

        self.suffixes = parameters.get('aliases', '_l,_r')
        self.suffixes = [s for s in self.suffixes.replace(" ", "").split(',')]
        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(
                                                 self.order))

    def generate_code(self):
        if self.has_code:
            
            if self.named_inputs['input data 1'] == self.named_inputs['input data 2']:
                copy = ".copy()"
            else:
                copy = ""

            code = """
            second_input = {in2}{copy}
            cols1 = [ '{suf_l}' + c for c in {in1}.columns]
            cols2 = [ '{suf_r}' + c for c in second_input.columns]
            
            {in1}.columns = cols1
            second_input.columns = cols2
            
            keys1 = ['{suf_l}' + c for c in {keys1}]
            keys2 = ['{suf_r}' + c  for c in {keys2}]
            
            for c in cols1:
                if isinstance({in1}[c].dtype, pd.Int64Dtype):
                    {in1}[c] = {in1}[c].astype(float)
            
            for c in cols2:
                if isinstance(second_input[c].dtype, pd.Int64Dtype):
                    second_input[c] = second_input[c].astype(float)
                    
            """.format(copy=copy,
                       in1=self.named_inputs['input data 1'],
                       in2=self.named_inputs['input data 2'],
                       suf_l=self.suffixes[0], suf_r=self.suffixes[1],
                       keys1=self.left_attributes, keys2=self.right_attributes)

            # Should be positive boolean logic? ---> '''if self.match_case:'''
            if not self.match_case:
                code += """
            data1_tmp = {in1}[keys1].apply(lower)
            data1_tmp.columns = [c + "_lower" for c in data1_tmp.columns]
            col1 = list(data1_tmp.columns)
            data1_tmp = cudf.concat([{in1}, data1_tmp], axis=1, sort=False)
                
            data2_tmp = second_input[keys2].apply(lower)
            data2_tmp.columns = [c + "_lower" for c in data2_tmp.columns]
            col2 = list(data2_tmp.columns)
            data2_tmp = cudf.concat([second_input, data2_tmp], axis=1, sort=False)
                    
            {out} = cudf.merge(data1_tmp, data2_tmp, left_on=col1, right_on=col2,
                copy=False, suffixes={suffixes}, how='{type}')
            # Why drop col_lower?
            {out}.drop(col1+col2, axis=1, inplace=True)
                 """.format(out=self.output, type=self.join_type,
                            in1=self.named_inputs['input data 1'],
                            id1=self.left_attributes,
                            id2=self.right_attributes,
                            suffixes=self.suffixes)
            else:
                code += """
                    
            {out} = cudf.merge({in1}, second_input, how='{type}', 
                    suffixes={suffixes},
                    left_on=keys1, right_on=keys2)
                 """.format(out=self.output, type=self.join_type,
                            in1=self.named_inputs['input data 1'],
                            suffixes=self.suffixes)

            if self.not_keep_right_keys:
                code += """
            cols_to_remove = keys2
            {out}.drop(cols_to_remove, axis=1, inplace=True)
                """.format(out=self.output)

            return dedent(code)


class TransformationOperation(Operation):
    """
    Returns a new DataFrame applying the expression to the specified column.
    Parameters:
        - Alias: new column name. If the name is the same of an existing,
            replace it.
        - Expression: json describing the transformation expression
    """
    ALIAS_PARAM = 'alias'
    EXPRESSION_PARAM = 'expression'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = any([len(self.named_inputs) > 0,
                             self.contains_results()])
        self.imports = set()
        if self.has_code:
            if self.EXPRESSION_PARAM in parameters:
                self.expressions = parameters[self.EXPRESSION_PARAM]
            else:
                msg = _("Parameter must be informed for task {}.")
                raise ValueError(
                    msg.format(self.EXPRESSION_PARAM, self.__class__))
            self.output = self.named_outputs.get(
                'output data', 'sampled_data_{}'.format(self.order))

    def generate_code(self):
        # Builds the expression and identify the target column
        params = {'input': self.named_inputs['input data']}
        functions = ""
        for expr in self.expressions:
            expression = expr['tree']
            expression = Expression(expression, params)
            f = expression.parsed_expression
            functions += "['{}', {}],".format(expr['alias'], f)

            self.imports.update(expression.imports)
            # row.append(expression.imports) #TODO: by operation itself

        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        code = """
        # FIX: In some cases, when a DataFrame is projected or filtered,
        # the Pandas BlockManager (internals Pandas structure) is not 
        # updated, slowing down significantly the performance of an 
        # apply operation.

        {out} = {input}{copy_code}
        functions = [{expr}]
        for col, function in functions:
            {out}[col] = {out}.apply(function, axis=1)
        """.format(copy_code=copy_code,
                   out=self.output, 
                   input=self.named_inputs['input data'],
                   expr=functions)
        return dedent(code)




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


class UnionOperation(Operation):
    """
    Return a new DataFrame containing all rows in this frame and another frame.
    Takes no parameters.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 2 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        if self.has_code:
            code = """
            {0} = cudf.concat([{1}, {2}], sort=False, axis=0, ignore_index=True)
            """.format(self.output,
                       self.named_inputs['input data 1'],
                       self.named_inputs['input data 2'])
            return dedent(code)
