# -*- coding: utf-8 -*-

import json

from juicer.spark.feature_operation import MaxAbsScalerOperation, StandardScalerOperation, MinMaxScalerOperation

class StandardScalerSparkOperation(StandardScalerOperation):

    def __init__(self, parameters, named_inputs, named_outputs):
        StandardScalerOperation.__init__(self, parameters, named_inputs, named_outputs)
        self.scaler = StandardScalerOperation(parameters, named_inputs, named_outputs)
        self.named_inputs = named_inputs
        self.parameters = parameters
        self.named_outputs = named_outputs

    def generate_code(self):
        input_data = self.named_inputs['input data']

        aliases = self.parameters["alias"].split(",")

        self.scaler.scaled_attr = "tmp_scaler"
        intern_code = self.scaler.generate_code()
        code = intern_code + '{output} = {output}.withColumn("tmp_scaler", vector_to_array("tmp_scaler")).select("*", *[functions.col("tmp_scaler").getItem(i).alias(c) for i, c in enumerate({aliases})]).drop("tmp_scaler")'.format(
            features=json.dumps(self.attribute),
            intern_code=intern_code,
            aliases=aliases,
            input=input_data,
            output=self.output ,)

        return code

class MinMaxScalerSparkOperation(MinMaxScalerOperation):

    def __init__(self, parameters, named_inputs, named_outputs):
        MinMaxScalerOperation.__init__(self, parameters, named_inputs, named_outputs)
        self.scaler = MinMaxScalerOperation(parameters, named_inputs, named_outputs)
        self.named_inputs = named_inputs
        self.parameters = parameters
        self.named_outputs = named_outputs

    def generate_code(self):
        input_data = self.named_inputs['input data']

        aliases = self.parameters["alias"].split(",")

        self.scaler.scaled_attr = "tmp_scaler"
        intern_code = self.scaler.generate_code()
        code = intern_code + '{output} = {output}.withColumn("tmp_scaler", vector_to_array("tmp_scaler")).select("*", *[functions.col("tmp_scaler").getItem(i).alias(c) for i, c in enumerate({aliases})]).drop("tmp_scaler")'.format(
            features=json.dumps(self.attribute),
            intern_code=intern_code,
            aliases=aliases,
            input=input_data,
            output=self.output ,)

        return code


class MaxAbsScalerSparkOperation(MaxAbsScalerOperation):

    def __init__(self, parameters, named_inputs, named_outputs):
        MaxAbsScalerOperation.__init__(self, parameters, named_inputs, named_outputs)
        self.scaler = MaxAbsScalerOperation(parameters, named_inputs, named_outputs)
        self.named_inputs = named_inputs
        self.parameters = parameters
        self.named_outputs = named_outputs

    def generate_code(self):
        input_data = self.named_inputs['input data']

        aliases = self.parameters["alias"].split(",")

        self.scaler.scaled_attr = "tmp_scaler"
        intern_code = self.scaler.generate_code()
        code = intern_code + '{output} = {output}.withColumn("tmp_scaler", vector_to_array("tmp_scaler")).select("*", *[functions.col("tmp_scaler").getItem(i).alias(c) for i, c in enumerate({aliases})]).drop("tmp_scaler")'.format(
            features=json.dumps(self.attribute),
            intern_code=intern_code,
            aliases=aliases,
            input=input_data,
            output=self.output ,)

        return code