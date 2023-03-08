import json
import os
import time
import re
import pdb
import numpy as np
import requests
import json
import yaml
import sys
import pandas as pd
import datetime
import copy
from juicer.multi_platform.limonero_api import Dataset
from juicer.multi_platform.operations import AddRowsOperationModel, AggregationOperationModel, \
    AddColumnsOperationModel, SortOperationModel, ProjectionOperationModel, ReplaceValueOperationModel,\
    CleanMissingOperationModel, SVMClassificationOperationModel, TransformationOperationModel, FilterSelectionOperationModel

df1 = Dataset(3391).stats

print(df1)
# print(df1.columns)


def _test_add_rows():
    print("# _test_add_rows")
    op1 = AddRowsOperationModel({"operation_id": 1})
    out = op1.estimate_output([df1, df1])
    print(df1)
    print(out[0])
    print(out[0].columns)


def _test_aggregation():
    print("# _test_aggregation")
    op1 = AggregationOperationModel({"operation_id": 1, "attributes": ["pclass", "age"],
                                     "function": [{'attribute': 'passengerid',
                                                   'f': 'AVG',
                                                   'alias': 'avg_passengerid'}]})
    out = op1.estimate_output([df1])
    print(df1)
    print(out[0])
    print(out[0].columns)


def _test_add_columns():
    print("# _test_add_columns")
    op1 = AddColumnsOperationModel({"operation_id": 1})
    out = op1.estimate_output([df1, df1])
    print(df1)
    print(out[0])
    print(out[0].columns)


def _test_sort():
    print("# _test_sort")
    op1 = SortOperationModel({"operation_id": 1, "attributes": ["pclass", "age"]})
    out = op1.estimate_output([df1])
    print(df1)
    print(out[0])
    print(out[0].columns)
    print(op1.gen_model())


def _test_projection():
    print("# _test_projection")
    op1 = ProjectionOperationModel({"operation_id": 1, "attributes": ["pclass", "age"]})
    out = op1.estimate_output([df1])
    print(df1)
    print(out[0])
    print(out[0].columns)
    print(op1.gen_model())


def _test_replace_values():
    print("# _test_replace_values")
    op1 = ReplaceValueOperationModel({"operation_id": 1})
    out = op1.estimate_output([df1])
    print(df1)
    print(out[0])
    print(out[0].columns)
    print(op1.gen_model())


def _test_clean_missing():
    print("# _test_clean_missing")
    op1 = CleanMissingOperationModel({"operation_id": 1, "cleaning_mode": "REMOVE_ROW"})
    out = op1.estimate_output([df1])
    print(df1)
    print(out[0])
    print(out[0].columns)
    print(op1.gen_model())


def _test_svm():
    print("# _test_svm")
    op1 = SVMClassificationOperationModel({"operation_id": 1,
                                           "attributes": ["pclass", "age"],
                                           "alias": "prediction"})
    out = op1.estimate_output([df1])
    print(df1)
    print(out[0])
    print(out[0].columns)
    print(op1.gen_model())


def _test_transformation():
    print("# _test_transformation")
    op1 = TransformationOperationModel({"operation_id": 1, 
    "expression": {"value": [
        {"alias": "new_age", 
        "expression": "age + 1", 
        "error": None, 
        "tree": {"type": "BinaryExpression", "operator": "+", "left": {"type": "Identifier", "name": "age"}, "right": {"type": "Literal", "value": 1, "raw": "1"}}}]}})
    out = op1.estimate_output([df1])
    print(df1)
    print(out[0])
    print(out[0].columns)
    print(op1.gen_model())

    
def _test_filter():
    print("# _test_filter")
    op1 = FilterSelectionOperationModel({"operation_id": 1, 
    "expression": {"value": [
        {"alias": "", 
        "expression": "age > 32", 
        "error": None, 
        "tree": {"type": "BinaryExpression", "operator": ">", "left": {"type": "Identifier", "name": "age"}, "right": {"type": "Literal", "value": 32, "raw": "32"}}}]}})
    
    print("Input: ", df1)
    out = op1.estimate_output([df1])
    
    print("Output: ",out[0])
    print("Columns: ", out[0].columns)
    print("Model: ", op1.gen_model()) 
    return out[0]