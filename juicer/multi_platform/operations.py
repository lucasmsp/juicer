# -*- coding: utf-8 -*-

from juicer.multi_platform.limonero_api import BaseStatistics
import copy
import json
from bisect import bisect
import datetime




class OperationModeling(object):

    
    def __init__(self):
        

        self.all_operations = {
            # etl
            "add-columns": AddColumnsOperationModel,  # OK
            "add-rows": AddRowsOperationModel,  # OK
            "aggregation": AggregationOperationModel,  # ok
            "difference": DifferenceOperationModel, # TODO
            "set-intersection": SetIntersectionOperationModel, # TODO
            "projection": ProjectionOperationModel,  # ok
            "drop": NonSupportedOperation,
            "remove-duplicated-rows": RemoveDuplicatedRowsOperationModel,  # ok
            "distinct": RemoveDuplicatedRowsOperationModel,  # ok
            "replace-value": ReplaceValueOperationModel, # ok
            "sort": SortOperationModel,  # ok
            "clean-missing": CleanMissingOperationModel, # ok
            "data-reader": DataReaderOperationModel,  # ok
            "data-writer": DataWriterOperationModel,  # ok
            "data-migration": DataMigrationOperationModel,
            "filter": FilterSelectionOperationModel,  # ok
            "filter-selection": FilterSelectionOperationModel,  # ok
            "transformation": TransformationOperationModel,  # ok
            "join": JoinOperationModel, # ok
            "remove-stop-words": RemoveStopWordsOperationModel, 
            "sample": SampleOperationModel, # ok
            "split": SplitOperationModel,  # ok
            "cast": ChangeAttributeOperationModel,
            
            # visualization
            "area-chart": VisualizationOperationModel,
            "bar-chart": VisualizationOperationModel,
            "box-plot": VisualizationOperationModel,
            "bubble-chart": VisualizationOperationModel,
            "donut-chart": VisualizationOperationModel,
            "heatmap": VisualizationOperationModel,
            "histogram": VisualizationOperationModel,
            "line-chart": VisualizationOperationModel,
            "map-chart": VisualizationOperationModel,
            "pie-chart": VisualizationOperationModel,
            "table-visualization": VisualizationOperationModel,  # ok
            "treemap": VisualizationOperationModel,
            "scatter-plot": VisualizationOperationModel,
            "summary-statistics": SummaryStatisticsOperationModel,  # ok
            "word-cloud": VisualizationOperationModel,
            
            # ml
            "apply-model": ApplyModelOperationModel,
            "outlier-detection": NonSupportedOperation,  # ok
            "feature-assembler": FeatureAssemblerOperationModel,
            "feature-indexer": FeatureIndexerOperationModel,  # ok
            "one-hot-encoder": OneHotEncoderOperationModel,    
            "tokenizer": TokenizerOperationModel,         
            "agglomerative-clustering": AgglomerativeClusteringOperationModel,
            "association-rules": AssociationRulesOperationModel,
            "bucketizer": KbinsDiscretizerOperationModel,
            "dbscan-clustering": DBSCANClusteringOperationModel,
            "decision-tree-classifier-model": DecisionTreeClassifierOperationModel,
            "evaluate-model": EvaluateModelOperationModel,
            "feature-disassembler": FeatureDisassemblerOperationModel,
            "frequent-item-set": FrequentItemSetOperationModel,
            "gaussian-mixture": GaussianMixtureClusteringOperationModel,
            'gaussian-mixture-clustering-model': GaussianMixtureClusteringOperationModel,
            'gbt-classifier': GBTClassifierOperationModel,
            "gbt-classifier-model": GBTClassifierOperationModel,
            "gbt-regressor-model": GBTRegressorOperationModel,
            'gbt-regressor': GBTRegressorOperationModel,
            'generalized-linear-regressor-model': GeneralizedLinearRegressor,
            'generalized-linear-regression': GeneralizedLinearRegressor,
            "huber-regressor-model": HuberRegressorOperationModel,
            'isotonic-regression-model': IsotonicRegressionOperationModel,
            'k-means-clustering-model': KMeansClusteringOperationModel,
            'k-means-clustering': KMeansClusteringOperationModel,
            'kbins-discretizer': KbinsDiscretizerOperationModel,
            "knn-classifier-model": KNNClassifierModelOperationModel,
            "lda-clustering-model": LDAClusteringOperationModel,
            "linear-regression-model": LinearRegressionOperationModel,
            'linear-regression': LinearRegressionOperationModel,
            'logistic-regression': LogisticRegressionOperationModel,
            'logistic-regression-model': LogisticRegressionOperationModel,
            #"logistic-regression-classifier-model": LogisticRegressionOperationModel,
            'max-abs-scaler': MaxAbsOperationModel,
            'min-max-scaler': MinMaxOperationModel,
            "mlp-regressor-model": MLPRegressorOperationModel,
            "mlp-classifier-model": MLPClassifierOperationModel,
            "naive-bayes-classifier-model": NaiveBayesClassificationOperationModel,
            "pca": PCAOperationModel,
            'perceptron-classifier-model': PerceptronClassifierOperationModel,
            "generate-n-grams": GenerateNGramsOperationModel,
            "quantile-discretizer": QuantileDiscretizerOperationModel,
            'random-forest-classifier-model': RandomForestClassifierOperationModel,
            "random-forest-regressor-model": RandomForestRegressorOperationModel,
            "regression-model": RegressorModelOperationModel,
            "sequence-mining": SequenceMiningOperationModel,
            "sgd-regressor-model": SGBRegressorOperationModel,
            'standard-scaler': StandardScalerOperationModel,
            "stdbscan": STDBSCANOperationModel,
            "split-k-fold": SplitKFoldOperationModel,
            "svm-classification-model": SVMClassificationOperationModel,  # ok
            "word-to-vector": WordToVectorOperationModel,
            
            # removed
            "execute-python": NonSupportedOperation,  # ok
            "execute-sql": NonSupportedOperation,  # ok

        }

        self.operations = self.all_operations
        self.slugs_dict = self._get_all_cleaned_slugs()
        self.slugs = sorted(list(self.slugs_dict.keys()))

    # def get_operation_pattern(self, slug):
    #     if slug in self.simple_operations:
    #         return "SIMPLE_PATTERN"
    #     elif slug in self.operations_only_features:
    #         return "ONLY_FEATURE_PATTERN"
    #     else:
    #         return "COMPLEX_PATTERN"

    def _get_all_cleaned_slugs(self):
        slugs = {s: self.all_operations[s].__name__ for s in self.all_operations.keys()}
        inv_map = {}
        for k, v in slugs.items():
            inv_map[v] = inv_map.get(v, []) + [k]
        return inv_map

    def get_operation(self, k):
        s = self.slugs_dict[k][0]
        return self.all_operations[s]()

    def clean_slug(self, slug):
        return self.all_operations[slug].__name__

    def get_all_dimensions_dict(self):
        slugs_fields = {"{}_{}".format(slug, field): 0
                        for slug in self.slugs
                        for field in self.get_operation(slug).features}
        return slugs_fields


class OperationModel(object):

    PHYSICAL_BEHAVIOR_READ = "read-operation"
    PHYSICAL_BEHAVIOR_WRITE = "write-operation"

    PHYSICAL_BEHAVIOR_VISUALIZATION = "visualization-operation"

    PHYSICAL_BEHAVIOR_SERIAL_EQ = "naive-serial-equal-rows"
    PHYSICAL_BEHAVIOR_SERIAL_LESS = "naive-serial-reduce-rows"
    PHYSICAL_BEHAVIOR_SERIAL_GREATER = "naive-serial-increase-rows"

    PHYSICAL_BEHAVIOR_WIDE_INFO_EQ = "wide-need-consensus-equal-size"
    PHYSICAL_BEHAVIOR_WIDE_INFO_LESS = "wide-need-consensus-reduce-size"

    PHYSICAL_BEHAVIOR_HASH_EQ = "wide-hash-partitioning-equal-size"
    PHYSICAL_BEHAVIOR_HASH_REDUCE = "wide-hash-partitioning-reduce-size"
    PHYSICAL_BEHAVIOR_RANGE = "wide-range-partitioning"

    PHYSICAL_BEHAVIOR_ML = "complex-ml"

    PLATFORM_SPARK = 1
    PLATFORM_PANDAS = 4
    PLATFORM_CUDF  = 6
    
    debug = False
    
    def __init__(self, parameters):
        self.features = None
        
        self.behavior = None
        self.parameters = parameters
        # self.platform_base = 1 if self.parameters["operation_id"] < 1000 else 4
        self.platform_target = self.get_platform_target()

    def convert(self, parameters):
        pass

    def get_cleaned_slug(self):
        return self.__name__

    def stage_behavior(self):
        if self.behavior in [self.PHYSICAL_BEHAVIOR_SERIAL_EQ,
                             self.PHYSICAL_BEHAVIOR_SERIAL_LESS,
                             self.PHYSICAL_BEHAVIOR_SERIAL_GREATER,
                             self.PHYSICAL_BEHAVIOR_READ,
                             self.PHYSICAL_BEHAVIOR_WIDE_INFO_EQ,
                             self.PHYSICAL_BEHAVIOR_WIDE_INFO_LESS]:
            return "same-stage"
        else:
            return "split-stage"


    def get_platform_target(self):
        platform_id = int(self.parameters["task"]["forms"].get("comment", {"value": -1})["value"])
        # if "task" in self.parameters:
        #    platform_id = int(self.parameters["task"]["forms"].get("comment", {"value": -1})["value"])
        # else:
        #     platform_id = -1

        return platform_id

    def convert(self, platform_id=None):
        pass

    def estimate_output(self, base_statistics):
        pass
        
    def gen_model(self, platform_target=None):
        return {}

    def extract_field_value(self, parameters, field):
        if "value" in parameters.get(field, {}):
            parameters[field] = parameters[field]["value"]
        return parameters
    
        

class GenericOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = None

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        return parameters


class AddColumnsOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        
        self.behavior = self.PHYSICAL_BEHAVIOR_HASH_EQ
        self.n_input = 1
        self.n_output = 1
        self.input = None
        self.output = None
        self.n_functions = 0

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("alias", {}):
            parameters["alias"] = parameters["alias"]["value"]
        return parameters

    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics)
        self.total_input_size_bytes_memory = self.input[0].size_bytes_memory + self.input[1].size_bytes_memory
        self.total_input_rows = self.input[0].n_rows + self.input[1].n_rows
        n1 = self.input[0].n_rows
        n2 = self.input[1].n_rows
        n_rows = max([n1, n2])

        self.output = BaseStatistics(n_rows=n_rows)
        diff = n_rows - n1
        for k, v in self.input[0].columns.items():
            name = "ds0_" + k
            self.output.columns[name] = copy.deepcopy(v)
            self.output.columns[name].name = name
            self.output.columns[name].n_rows = n_rows
            if diff > 0:
                self.output.columns[name].missing_total += diff

        diff = n_rows - n2
        for k, v in self.input[1].columns.items():
            name =  "ds1_" + k
            self.output.columns[name] = copy.deepcopy(v)
            self.output.columns[name].name = name
            self.output.columns[name].n_rows = n_rows
            if diff > 0:
                self.output.columns[name].missing_total += diff

        self.output.recalculate()
        return [self.output]

    def gen_model(self, platform_target=None):
        return {
            #"input_size_bytes_memory": 
            #    self.input[0].size_bytes_memory + self.input[1].size_bytes_memory,
            "input_n_rows": self.input[0].n_rows + self.input[1].n_rows,
            "output_n_rows": self.output.n_rows,
            #'n_rows_columns': (self.input[0].n_rows + self.input[1].n_rows)
            #"n_columns": self.output.n_columns,
            "target_engine": self.platform_target
        }


class AddRowsOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        
        self.behavior = self.PHYSICAL_BEHAVIOR_SERIAL_EQ
        self.n_input = 2
        self.n_output = 1
        self.output = None
        self.input = None

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("alias", {}):
            parameters["alias"] = parameters["alias"]["value"]
        return parameters

    def estimate_output(self, base_statistics):
        # Assume que as os dois datasets possuem mesmas colunas
        self.input = copy.deepcopy(base_statistics)
        self.total_input_size_bytes_memory = self.input[0].size_bytes_memory + self.input[1].size_bytes_memory
        self.total_input_rows = self.input[0].n_rows + self.input[1].n_rows
        n_rows = sum([r.n_rows for r in base_statistics])
        self.output = copy.deepcopy(base_statistics[0])
        self.output.n_rows = n_rows
        
        # self.input[0].print_all_columns()
        # self.input[1].print_all_columns()

        for key, value in base_statistics[0].columns.items():

            if key not in base_statistics[1].columns:
                raise Exception("DataFrame must have same columns")
            value2 = base_statistics[1].columns[key]

            values = {"n_rows": n_rows,
                      "distinct_values": None,
                      "median_value": None}

            if value.min_value and value2.min_value:
                values["min_value"]: min([value.min_value, value2.min_value])

            if value.max_value and value2.max_value:
                values["max_value"]: min([value.max_value, value2.max_value])

            if value.missing_total and value2.missing_total:
                values["missing_total"]: sum([value.missing_total, value2.missing_total])

            if value.distinct_values and value2.distinct_values:
                values["distinct_values"] = max([value.distinct_values, value2.distinct_values])

            self.output.update_column(key, values)

        self.output.recalculate()
        return [self.output]

    def gen_model(self, platform_target=None):

        return {
             "input_n_rows": self.input[0].n_rows + self.input[1].n_rows,
            "output_n_rows": self.output.n_rows,
            # "input_size_bytes_memory": 
            #     self.input[0].size_bytes_memory + self.input[1].size_bytes_memory,
            #"output_size_bytes_memory": self.output.size_bytes_memory,
            "n_columns": self.output.n_columns,
            "target_engine": self.platform_target
        }


class AgglomerativeClusteringOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML
        self.n_input = 1
        self.n_output = 1


class AggregationOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        
        self.behavior = self.PHYSICAL_BEHAVIOR_HASH_REDUCE
        self.n_input = 1
        self.n_output = 1
        self.input = None
        self.output = None
        self.n_functions = 0

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("alias", {}):
            parameters["alias"] = parameters["alias"]["value"]
            parameters["attributes"] = parameters["attributes"]["value"]
            parameters["function"] = parameters["function"]["value"]
        return parameters

    def estimate_output(self, base_statistics):
        parameters = self.convert()
        keys = parameters["attributes"]
        
        self.input = copy.deepcopy(base_statistics[0])
        #print(self.input)
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows
        distinct_values = []

        for k in keys:
            distinct_values.append(self.input.columns[k].distinct_values)

        distinct_values = max(distinct_values)

        
        self.output = BaseStatistics(n_rows=distinct_values)

        for k in keys:
            self.output.columns[k] = copy.deepcopy(base_statistics[0].columns[k])
            self.output.update_column(k, {"n_rows": distinct_values, "missing_total": 0})
            if self.output.columns[k].deciles:
                self.output.columns[k].deciles = {k: 1 for k in self.output.columns[k].deciles}
            
            
        for row in parameters["function"]:
            alias = row["alias"]
            att = row["attribute"]
            f = row['f']
            _type = 'INTEGER'
            
            
            if f in ['count', 'size']:
                _type = 'INTEGER'
                min_value = 0
                max_value = 999999
                att_distinct_values = distinct_values
            elif f in ['sum', 'max', 'min', 'first', 'last', 'avg', 'mean']:
                _type = self.input.columns[att].type
                min_value = self.input.columns[att].min_value
                max_value = self.input.columns[att].max_value
                att_distinct_values = min([self.input.columns[att].distinct_values, distinct_values])
            elif f in ['countDistinct']:
                _type = 'INTEGER'
                min_value = 0
                max_value = 999999
                att_distinct_values = min([self.input.columns[att].distinct_values, distinct_values])
            else:
                raise Exception("Aggregation function not supported")

            self.output.create_new_column(**{
                "n_rows": distinct_values,
                "name": alias,
                "type": _type,
                "missing_total": 0,
                'min_value': min_value,
                'max_value': max_value,
                'distinct_values': att_distinct_values,
                
                
            })
            self.n_functions += 1

        self.output.recalculate()
        if self.debug:
            print("AggregationOperationModel | ", self.output)
            print("AggregationOperationModel | columns:", self.output.columns)
            for c in self.output.columns:
                print(f"AggregationOperationModel | deciles column {c}:", self.output.columns[c].deciles)
        return [self.output]

    def gen_model(self, platform_target=None):
        return {
            "input_n_rows": self.input.n_rows,
            "output_n_rows": self.output.n_rows,
            # "input_size_bytes_memory": self.input.size_bytes_memory,
            # "output_size_bytes_memory": self.output.size_bytes_memory,
            "data_ratio": self.input.n_rows / self.output.n_rows,
            "n_functions": self.n_functions,
            "target_engine": self.platform_target
        }



class ApplyModelOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_SERIAL_EQ
        self.col_label_trained = None

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()

        parameters = self.extract_field_value(parameters, "features")
        parameters = self.extract_field_value(parameters, "prediction")
        
        self.col_label_trained = parameters['task']['col_label_trained']['label']['value'][0]
        return parameters
    
    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows

        parameters = self.convert()
        self.label = parameters['prediction']
        
        if self.col_label_trained:
                self.output.create_new_column(**{
                "name": self.label,
                "type": "DOUBLE",
                "missing_total": 0,
                'distinct_columns': self.col_label_trained.distinct_values,
                'min_value': self.col_label_trained.min,
                'max_value': self.col_label_trained.max
            })
        else:
            self.output.create_new_column(**{
                "name": self.label,
                "type": "DOUBLE",
                "missing_total": 0,
                'distinct_columns': self.total_input_rows*0.10,
                'min_value': 0,
                'max_value': 1
            })
        self.output.recalculate()

        return [self.output]


class AssociationRulesOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, parameters):
        pass


class ChangeAttributeOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.n_input = 1
        self.n_output = 1
        self.input = None
        self.output = None
        self.n_functions = 0

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("alias", {}):
            parameters["alias"] = parameters["alias"]["value"]
        return parameters


class CleanMissingOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        
        self.n_input = 1
        self.n_output = 1
        self.input = None
        self.output = None
        self.n_subset = None
        self.mode = self.parameters.get("cleaning_mode", "REMOVE_ROW")

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("attributes", {}):
            parameters["attributes"] = parameters["attributes"]["value"]
        return parameters

    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows
        self.output = copy.deepcopy(base_statistics[0])
        parameters = self.convert()

        if "attributes" in parameters:
            keys = parameters["attributes"]
        else:
            keys = list(self.output.columns.keys())

        self.n_subset = len(keys)
        
        if self.mode == "REMOVE_ROW":
            max_col = ""
            n_max_row = 0
            for c in keys:
                missing = self.output.columns[c].missing_total
                if missing > n_max_row:
                    n_max_row = missing
                    max_col = c

            if max_col != "":
                for c in self.output.columns:
                    self.output.columns[c].missing_total = 0
                    self.output.columns[c].n_rows -= n_max_row
            self.output.remove_random_deciles(n_max_row, except_list=[max_col])
            
        elif self.mode == "REMOVE_COLUMN":
            for c in keys:
                if self.output.columns[c].missing_total > 0:
                    del self.output.columns[c]
        else:
            for c in self.output.columns:
                self.output.columns[c].missing_total = 0
        
        self.output.recalculate()
        if self.debug:
            print("CleanMissingOperationModel | output: ", self.output)
            print("CleanMissingOperationModel | ", self.output.columns)
            for c in self.output.columns:
                print(f"CleanMissingOperationModel | {c}", self.output.columns[c].deciles)
      
        return [self.output]

    def gen_model(self, platform_target=None):

        return {
            #"input_size_bytes_memory": self.input.size_bytes_memory,
            #"output_size_bytes_memory": self.output.size_bytes_memory,
            "input_n_rows": self.input.n_rows,
            "output_n_rows": self.output.n_rows,
            "reduced_ratio": self.input.n_rows / self.output.n_rows,
            "n_attributes": self.n_subset,
            "target_engine": self.platform_target
        }


class DataReaderOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        
        self.behavior = self.PHYSICAL_BEHAVIOR_READ
        self.n_input = 0
        self.n_output = 1
        self.output = None

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("alias", {}):
            parameters["alias"] = parameters["alias"]["value"]
        return parameters

    def estimate_output(self, base_statistics):
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = 0
        self.total_input_rows = self.output.n_rows
        return [self.output]

    def gen_model(self, platform_target=None):
        return {
            "output_n_rows": self.output.n_rows,
            #"size_megabytes_disk": self.output.size_megabytes_disk,
            #"output_size_bytes_memory": self.output.size_bytes_memory,
            "n_columns": self.output.n_columns,
            "target_engine": self.platform_target
        }


class DataWriterOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        #elf.data_amplification = 1.0
        self.behavior = self.PHYSICAL_BEHAVIOR_WRITE
        self.n_input = 0
        self.n_output = 1
        self.output = None

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("alias", {}):
            parameters["alias"] = parameters["alias"]["value"]
        return parameters

    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.output.size_bytes_memory
        self.total_input_rows = self.input.n_rows
        return [self.output]

    def gen_model(self, platform_target=None):
        return {
            "input_n_rows": self.input.n_rows,
            "n_columns": self.output.n_columns,
            #n_rows_plus_columns': self.output.n_rows * self.output.n_columns,
            "target_engine": self.platform_target
        }

    
class DataMigrationOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        
        self.behavior = self.PHYSICAL_BEHAVIOR_WRITE
        self.n_input = 0
        self.n_output = 1
        self.output = None

        self.features = {
            'source_engine': int(parameters.get("source_engine", -1))
        }


    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        return parameters

    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows
        return [self.output]

    def gen_model(self, platform_target=None):
        return {
            #"input_size_bytes_memory": self.output.size_bytes_memory,
            #'target_platform': self.platform_target,
            "n_rows": self.output.n_rows,
            #"n_columns": self.output.n_columns,
            # "platform_id": self.platform_target,
            'source_engine': self.features['source_engine'],
            'target_engine': self.platform_target,
        }
    

class DBSCANClusteringOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        #  self.behavior = self.PHYSICAL_BEHAVIOR_ML
        self.behavior = "not-supported"


class DecisionTreeClassifierOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, parameters):
        pass


class DifferenceOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        
        self.behavior = self.PHYSICAL_BEHAVIOR_HASH_REDUCE


class EvaluateModelOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class NonSupportedOperation(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = "not-supported"


class FeatureAssemblerOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_SERIAL_EQ

    def convert(self, parameters):
        print("FeatureAssemblerOperationModel:", parameters)
        if "value" in parameters.get("alias", {}):
            parameters["alias"] = parameters["alias"]["value"]
        return parameters


class FeatureDisassemblerOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_SERIAL_EQ

    def convert(self, parameters):
        print("FeatureDisassemblerOperationModel:", parameters)
        return parameters


class FeatureIndexerOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("attributes", {}):
            parameters["attributes"] = parameters["attributes"]["value"]

        if "alias" in parameters.get("alias", {}):
            parameters["alias"] = parameters["alias"]["value"]
        return parameters

    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows

        parameters = self.convert()
        self.features_col = parameters["attributes"]
        new_cols = parameters["alias"].split(",")

        for col, new_col in zip(self.features_col, new_cols):
            n_distinct_values = self.input.columns[col].distinct_values
            self.output.create_new_column(**{
                "name": new_col,
                "type": "DOUBLE",
                "missing_total": 0,
                'distinct_columns': n_distinct_values,
                'min_value': 0,
                'max_value': n_distinct_values
            })
        self.output.recalculate()

        return [self.output]
        
    def gen_model(self, platform_target=None):

        return {
            "n_rows": self.input.n_rows,
            "n_features": len(self.features_col),
            "target_engine": self.platform_target
        }

class FilterSelectionOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        
        self.behavior = self.PHYSICAL_BEHAVIOR_SERIAL_LESS
        self.n_input = 0
        self.n_output = 1
        self.output = None

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("attributes", {}):
            parameters["attributes"] = parameters["attributes"]["value"]
        
        if "value" in parameters.get("expression", {}):
            parameters["expression"] = parameters["expression"]["value"]
        return parameters
    
    def parser_empression(self, parameters):
        
        expressions = []
        for expression in parameters["expression"]:
            tree = expression["tree"]
            #print(tree)
            # {'type': 'BinaryExpression', 
            #  'operator': '<=', 
            # 'left': {'type': 'Identifier', 'name': 'shipdate'}, 
            # 'right': {'type': 'CallExpression', 'arguments': [{'type': 'Literal', 'value': '1998-09-16', 'raw': "'1998-09-16'"}], 'callee': {'type': 'Identifier', 'name': 'timestamp'}}}
            if tree["type"] == "BinaryExpression":
                ops = {"operator": tree["operator"]}
                
                for side in ["left", "right"]:
                    s_type = tree[side]["type"]
                    
                    if s_type == "Identifier":
                        s_value = tree[side]["name"]
                        ops["side"] = side
                    elif s_type == "Literal":
                        s_value = tree[side]["value"]
                    elif s_type == "CallExpression" :
                        s_type = "Literal"
                        s_value = tree[side]['arguments'][0]['value']
                        
                    ops[s_type] = s_value
            else:
                raise Exception("Only BinaryExpression expression is supported")
   
            order = ops["side"]
            operator = ops["operator"]

            if (order == "right") and (operator in [">", "<", ">=", "<="]):
                if ">" in operator:
                    operator = operator.replace(">", "<")
                else:
                    operator = operator.replace("<", ">")
                    
            expressions.append(ops)
        return expressions  

    
    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(self.input)
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows
        parameters = self.convert()
        
        #TODO: apenas uma expressão é suportada
        self.n_expressions = len(parameters["expression"])
        expressions = self.parser_empression(parameters)
        
        for ops in expressions:
            to_remove = 0
            col_name = ops["Identifier"]     
            order = ops["side"]
            operator = ops["operator"]
            
            if self.output.columns[col_name].deciles:
                
                value = ops["Literal"]
                deciles = self.output.columns[col_name].deciles
                min_value = self.output.columns[col_name].min_value
                max_value = self.output.columns[col_name].max_value
                
                if  self.output.columns[col_name].is_string():
                    if operator == "==":
                        for k in list(deciles.keys()):
                            if k != value:
                                to_remove += deciles[k]
                                del deciles[k]
                                
                    elif operator == "!=":
                        for k in list(deciles.keys()):
                            if k == value:
                                to_remove += deciles[k]
                                del deciles[k]
                                
                    self.output.columns[col_name].deciles = deciles  
                    
                
                elif self.output.columns[col_name].is_number() or self.output.columns[col_name].is_date():

                    if self.output.columns[col_name].is_date():
                        value = float(datetime.datetime.strptime(value, '%Y-%m-%d').strftime('%s'))
                        #print(value)
                    else:
                        value = float(value)

                    min_value = float(min_value)
                    max_value = float(max_value)

                    # if not (min_value <= value <= max_value):
                    #     self.output.remove_all_rows()
 
                    deciles_keys = sorted(list(deciles.keys()))

                    if operator == "==":
                        idx = bisect(list(deciles.keys()), value)

                        for i in range(idx, len(deciles_keys)):
                            key = deciles_keys[i]
                            if idx != i:
                                to_remove += deciles[key]
                                del deciles[key]

                    elif operator == "!=":

                        idx = bisect(list(deciles.keys()), value)

                        for i in range(idx, len(deciles_keys)):
                            key = deciles_keys[i]
                            if idx == i:
                                to_remove += deciles[key]
                                del deciles[key]                                

                    else:
                        # descobrindo em qual idx do deciles o literal está
                        if value in deciles:
                            idx = list(deciles.keys()).index(value)
                        else:
                            idx = bisect(list(deciles.keys()), value)

                        #print(f"Literal {value} | IDX: {idx}")

                        to_remove = self.output.columns[col_name].missing_total
                        if ">" in operator:
                            for i in range(0, idx):
                                key = deciles_keys[i]
                                #print(f' Removing >: {key} --> {deciles[key]}')
                                to_remove += deciles[key]
                                del deciles[key]
                        else:
                            for i in range(idx+1, len(deciles_keys)):
                                key = deciles_keys[i]
                                #print(f' Removing <: {key} --> {deciles[key]}')
                                to_remove += deciles[key]
                                del deciles[key]

                    self.output.columns[col_name].deciles = deciles  
                    
            else:
                self.output.recalculate() 
                to_remove = int(self.output.n_rows * 0.25)
            
            self.output.remove_random_deciles(to_remove, except_list=[col_name])

        #print("original:", self.output.n_rows)
        #print(to_remove)
        self.output.remove_n_rows(to_remove) 
        self.output.recalculate() 
        #print("after:", self.output.n_rows)
        
        if self.debug:
            print(f"FilterSelectionOperationModel | ", self.output)
            self.output.print_all_columns()
            for c in self.output.columns:
                print(f"FilterSelectionOperationModel | {c}", self.output.columns[c].deciles)
#         print("----")
        #
#         print("----")
        
        return [self.output]

    def gen_model(self, platform_target=None):

        return {
            #"input_size_bytes_memory": self.input.size_bytes_memory,
            #"output_size_bytes_memory": self.output.size_bytes_memory,
            "input_n_rows": self.input.n_rows,
            "output_n_rows": self.output.n_rows,
            #"output_n_rows": self.output.n_rows,
            #"ratio": self.output.n_rows / self.input.n_rows,
            'n_expressions': self.n_expressions,
            "target_engine": self.platform_target
        }


class FrequentItemSetOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, parameters):
        if "value" in parameters.get('min_support', {}):
            parameters['min_support'] = parameters['min_support']['value']
            parameters['min_confidence'] = parameters['min_confidence']['value']
            parameters['attribute'] = parameters['attribute']['value']
        return parameters



class GaussianMixtureClusteringOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, parameters):
        pass


class GBTClassifierOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, parameters):
        pass


class GBTRegressorOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, parameters):
        pass


class GeneralizedLinearRegressor(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class HuberRegressorOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, parameters):
        pass


class IsotonicRegressionOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class JoinOperationModel(OperationModel):

    KEEP_RIGHT_KEYS_PARAM = 'keep_right_keys'
    MATCH_CASE_PARAM = 'match_case'
    JOIN_TYPE_PARAM = 'join_type'
    LEFT_ATTRIBUTES_PARAM = 'left_attributes'
    RIGHT_ATTRIBUTES_PARAM = 'right_attributes'

    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        
        self.behavior = self.PHYSICAL_BEHAVIOR_HASH_EQ

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in self.parameters["aliases"]:
            parameters["aliases"] = self.parameters["aliases"]["value"]
            parameters[self.MATCH_CASE_PARAM] = self.parameters[self.MATCH_CASE_PARAM]["value"]
            parameters[self.JOIN_TYPE_PARAM] = self.parameters[self.JOIN_TYPE_PARAM]["value"]
            parameters[self.LEFT_ATTRIBUTES_PARAM] = self.parameters[self.LEFT_ATTRIBUTES_PARAM]["value"]
            parameters[self.RIGHT_ATTRIBUTES_PARAM] = self.parameters[self.RIGHT_ATTRIBUTES_PARAM]["value"]
        return parameters
    
    def estimate_output(self, base_statistics):
        self.input1 = copy.deepcopy(base_statistics[0])
        self.input2 = copy.deepcopy(base_statistics[1])
        self.total_input_size_bytes_memory = self.input1.size_bytes_memory + self.input2.size_bytes_memory
        self.total_input_rows = self.input1.n_rows + self.input2.n_rows
        parameters = self.convert()
        # Por enquanto só considero KEEP_RIGHT_KEYS_PARAM como false
        cols_output = []

        suffixes = parameters.get('aliases', '_l,_r')
        suffixes = [s for s in suffixes.replace(" ", "").split(',')]
        
        cols1 = [ suffixes[0] + c for c in self.input1.columns]
        cols2 = [ suffixes[1] + c for c in self.input2.columns]
        
        left_attributes  = [ suffixes[0] + c for c in parameters.get(self.LEFT_ATTRIBUTES_PARAM)]
        right_attributes = [ suffixes[1] + c for c in parameters.get(self.RIGHT_ATTRIBUTES_PARAM)]
        self.n_keys = len(left_attributes)
        join_type = parameters.get(self.JOIN_TYPE_PARAM, 'inner')\
                .replace("_outer", "") 
        
        # print(left_attributes)
        #print(self.input1.columns)
        # print(self.input1.columns[parameters.get(self.LEFT_ATTRIBUTES_PARAM)[0]].deciles)
        # print(right_attributes)
        #print(self.input2.columns)
        # print(self.input2.columns[parameters.get(self.RIGHT_ATTRIBUTES_PARAM)[0]].deciles)

        confs_output = {}
        def getOverlap(a, b):
            a_s, a_e = a 
            b_s, b_e = b
            if (b_s > a_e) or (a_s > b_e):
                return None
            else:
                return max(a_s, b_s), min(a_e, b_e)

        if join_type == "left":
            n_rows = self.input1.n_rows
            self.join_type = 0
            

        elif join_type == "right":
            n_rows = self.input2.n_rows
            self.join_type = 0

        elif join_type == "inner":
            self.join_type = 1
            for l, r in zip(parameters.get(self.LEFT_ATTRIBUTES_PARAM)[:1], 
                            parameters.get(self.RIGHT_ATTRIBUTES_PARAM)[:1]):
                
                if self.input1.columns[l].is_number() or self.input1.columns[l].is_date():
                
                    interval_overlap = getOverlap(
                        [float(self.input1.columns[l].min_value), float(self.input1.columns[l].max_value)],
                        [float(self.input2.columns[r].min_value), float(self.input2.columns[r].max_value)]
                    )
                    confs_output['overlap'] = interval_overlap

                    
                    distinct_l = self.input1.columns[l].distinct_values
                    distinct_r = self.input2.columns[r].distinct_values
                    
                    distinct_factor_l = self.input1.columns[l].get_elements_interval(interval_overlap) /  distinct_l
                    distinct_factor_r = self.input2.columns[r].get_elements_interval(interval_overlap) / distinct_r
                    
                    n_rows =  (distinct_factor_l * distinct_factor_r) * min([distinct_l, distinct_r])
                    
                elif self.input1.columns[l].is_string():
                    deciles1 = self.input1.columns[l].deciles
                    deciles2 = self.input2.columns[r].deciles
                    inter = set(deciles1.keys()).intersection(set(deciles2.keys()))
                    
                    interval_overlap = [max([self.input1.columns[l].min_value, self.input2.columns[r].min_value]), 
                                        min([self.input1.columns[l].max_value, self.input2.columns[r].max_value])]
                    
                    confs_output['overlap'] = interval_overlap

                    
                    
                    # distinct_factor_l = (self.input1.columns[l].distinct_values)
                    # distinct_factor_r = (self.input2.columns[r].distinct_values) 
                    # n_rows =  max([distinct_factor_l, distinct_factor_r])
                    
                    distinct_l = self.input1.columns[l].distinct_values
                    distinct_r = self.input2.columns[r].distinct_values
                    
                    distinct_factor_l = self.input1.columns[l].get_elements_interval(interval_overlap) /  distinct_l
                    distinct_factor_r = self.input2.columns[r].get_elements_interval(interval_overlap) / distinct_r
                    
                    #n_rows =  (distinct_factor_l * distinct_factor_r) *  min([distinct_l, distinct_r])
                    n_rows = max([self.input1.columns[l].n_rows, self.input2.columns[r].n_rows])
                else:
            
                    distinct_factor_l = (self.input1.columns[l].n_rows - self.input1.columns[l].missing_total)
                    distinct_factor_r = (self.input2.columns[r].n_rows - self.input2.columns[r].missing_total) 

                    n_rows =  max([distinct_factor_l, distinct_factor_r])
                    
                # backup: 20230901
                #n_rows =  min([distinct_factor_l, distinct_factor_r])
                
                if self.debug:
                    print(f"JoinOperation | distinct_factor_l {distinct_factor_l} - distinct_factor_r {distinct_factor_r} - n_rows {n_rows} - ")

        else:
            raise Exception("Join Type not supported.")
        
        n_rows = int(n_rows)
        self.output = BaseStatistics(n_rows=n_rows)
        
        for i_col, o_col in zip(self.input1.columns, cols1):
            lemonade_col = self.input1.columns[i_col]
            lemonade_col.name = o_col
            lemonade_col.missing_total = 0  

            lemonade_col.update_rows(n_rows)
            self.output.columns[o_col] = lemonade_col

        new_min, new_max = confs_output['overlap']
        for o_col in left_attributes:
            self.output.columns[o_col].min_value = max([new_min, self.output.columns[o_col].min_value])
            self.output.columns[o_col].max_value = min([new_max, self.output.columns[o_col].max_value])


        for i_col, o_col in zip(self.input2.columns, cols2):
            lemonade_col = self.input2.columns[i_col]
            lemonade_col.name = o_col
            lemonade_col.missing_total = 0

            lemonade_col.update_rows(n_rows)
            self.output.columns[o_col] = lemonade_col

        for o_col in right_attributes:    
            new_min, new_max = confs_output['overlap']
            self.output.columns[o_col].min_value = max([new_min, self.output.columns[o_col].min_value])
            self.output.columns[o_col].max_value = min([new_max, self.output.columns[o_col].max_value])

        self.output.recalculate()
        if self.debug:
            print("JoinOperation | input 1")
            self.input1.print_all_columns()
            print("JoinOperation | input 2")
            self.input2.print_all_columns()
            print(f"JoinOperation |", self.output)
            self.output.print_all_columns()
        
        return [self.output]

    def gen_model(self, platform_target=None):
        ds = [v for v in [self.input1.size_bytes_memory, self.input2.size_bytes_memory] if v >= (128*(1024*1024))]
        join_implementation = len(ds)

        if platform_target == self.PLATFORM_PANDAS:
            
            return {
                "input_size": self.total_input_size_bytes_memory,
                "output_size": self.output.size_bytes_memory,
                "join_type": self.join_type,
                "target_engine": self.platform_target
            }
        else:
            return {
                "input_size": self.total_input_size_bytes_memory,
                "output_size": self.output.size_bytes_memory, 
                #"input_size_bytes_memory": self.input1.size_bytes_memory + self.input2.size_bytes_memory,
                #"output_size_bytes_memory": self.output.size_bytes_memory,
                'join-implementation': join_implementation,
                "join_type": self.join_type,
                # 'n_keys': self.n_keys,
                "target_engine": self.platform_target
            }


class KMeansClusteringOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML
        self.features = {}

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()

        if "value" in parameters.get("features", {}):
            parameters["features"] = parameters["features"]["value"]

        if "alias" in parameters:
            if "value" in parameters.get("alias", {}):
                parameters["alias"] = parameters["alias"]["value"]
        else:
            parameters["alias"] = "prediction"

        #print(parameters)
        if "value" in parameters.get("max_iterations", {}):
            parameters["max_iterations"] = parameters["max_iterations"]["value"]
        
        if "value" in parameters.get("number_of_clusters", {}):
            parameters["number_of_clusters"] = parameters["number_of_clusters"]["value"]
            
        if "value" in parameters.get("tolerance", {}):
            parameters['tolerance'] = parameters["tolerance"]['value']
            
        # to spark
        if int(platform_id) == 1:
            parameters["paramgrid"] = {"max_iterations": parameters["max_iterations"]}
            parameters['type'] = "kmeans"
        else:
            parameters['type'] = 'K-Means'
            parameters['n_clusters'] = parameters["number_of_clusters"]
            parameters['tol'] = parameters["tolerance"]
            parameters['max_iter'] = parameters["max_iterations"]
            
        self.features['k'] = int(self.parameters["number_of_clusters"])
        self.features["max_iter"] = int(self.parameters.get("max_iterations", 1000))
        
        return parameters

    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows

        platform_id = self.get_platform_target()
        parameters = self.convert(platform_id)
        self.features_col = parameters["features"]
        # if len(col_features) > 0:
        #     self.output.col_as_features
        
        v_per_k = self.total_input_rows//self.features['k']
        deciles = {}
        for i in range(self.features['k']):
            deciles[i] = v_per_k
        diff = self.total_input_rows - sum(deciles.values())
        deciles[0] += diff
        
        self.output.create_new_column(**{
            "name": parameters["alias"],
            "type": "DOUBLE",
            "missing_total": 0,
            'distinct_values': self.features['k'],
            'min_value': 0,
            'max_value': self.features['k'],
        })
        self.output.columns[parameters["alias"]].deciles = deciles
        self.output.recalculate()
        
        if self.debug:
            self.output.print_all_columns()
            for col in self.output.columns:
                print(f"KMeansClusteringOperationModel: col {col}:", self.output.columns[col].deciles)
        return [self.output]
    
    def gen_model(self, platform_target=None):
        
        if platform_target == self.PLATFORM_PANDAS:
            return {
                #v1
                "n_rows": self.input.n_rows,
                "max_iter":  self.features["max_iter"],
                'k': self.features['k'],
                "n_features": len(self.features_col),
                "target_engine": self.platform_target
            }
        else:
            return {
                #v1
                 "n_rows": self.input.n_rows,
                "max_iter":  self.features["max_iter"],
                'k': self.features['k'],
                "n_features": len(self.features_col),
                "target_engine": self.platform_target
            }


class KbinsDiscretizerOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class KNNClassifierModelOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

        #     #     parameters["n_neighbors"] = parameters["n_neighbors"]["value"]
        #     #     parameters["p"] = parameters["p"]["value"]
        #     #     parameters["algorithm"] = parameters["algorithm"]["value"]
        #     #     parameters["leaf_size"] = parameters["leaf_size"]["value"]


class LDAClusteringOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class LinearRegressionOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, platform_id=-1):
        parameters = self.parameters.copy()
        #print('LogisticRegressionOperationModel: ', parameters)

        parameters = self.extract_field_value(parameters, "features")
        parameters = self.extract_field_value(parameters, "label")
        parameters = self.extract_field_value(parameters, "prediction")
        parameters = self.extract_field_value(parameters, "elastic_net")   
        parameters = self.extract_field_value(parameters, "max_iter")  


        if int(platform_id) == self.PLATFORM_PANDAS:
            # to sklearn
            if "elastic_net" in parameters:
                parameters["alpha"] = parameters["elastic_net"]
        
        return parameters
    
    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows

        parameters = self.convert()
        self.features_col = parameters["features"]
        self.label = parameters['label'][0]
        self.features['max_iter'] = parameters.get("max_iter", 100)
        
        self.output.create_new_column(**{
            "name": parameters.get("prediction","prediction"),
            "type": "DOUBLE",
            "missing_total": 0,
            'distinct_columns': self.input.columns[self.label].distinct_values,
            'min_value': self.input.columns[self.label].min_value,
            'max_value': self.input.columns[self.label].max_value
        })
        self.output.recalculate()

        return [self.output]

    def gen_model(self, platform_target=None):
        
        return {
            #v1
            "n_rows": self.input.n_rows,
            "max_iter":  self.features["max_iter"],
            "n_features": len(self.features_col),
            "target_engine": self.platform_target
        }


class LogisticRegressionOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        #print('LogisticRegressionOperationModel: ', parameters)

        parameters = self.extract_field_value(parameters, "features")
        parameters = self.extract_field_value(parameters, "label")
        parameters = self.extract_field_value(parameters, "tol")
        parameters = self.extract_field_value(parameters, "regularization")   
        parameters = self.extract_field_value(parameters, "max_iter")  

        if int(platform_id) == 1:
            # to spark
            parameters["paramgrid"] = {"max_iter": parameters["max_iter"]}
            parameters["reg_param"] = parameters["regularization"]
        return parameters

    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows

        parameters = self.convert()
        self.features_col = parameters["features"]
        self.label = parameters['label']
        n_distinct_values = self.input.columns[self.label].distinct_values
        
        self.output.create_new_column(**{
            "name": parameters["alias"],
            "type": "DOUBLE",
            "missing_total": 0,
            'distinct_columns': n_distinct_values,
            'min_value': 0,
            'max_value': n_distinct_values
        })
        self.output.recalculate()

        return [self.output]


class MaxAbsOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML
        self.spark_convert_features = True

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()

        parameters = self.extract_field_value(parameters, "attributes")
        parameters = self.extract_field_value(parameters, "alias")

        parameters['attribute'] = parameters['attributes']

        return parameters

    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows

        parameters = self.convert()
        self.features_col = parameters["features"]
        self.label = parameters['label']
        n_distinct_values = self.input.columns[self.label].distinct_values
        
        self.output.create_new_column(**{
            "name": parameters["alias"],
            "type": "DOUBLE",
            "missing_total": 0,
            'distinct_columns': n_distinct_values,
            'min_value': 0,
            'max_value': n_distinct_values
        })
        self.output.recalculate()

        return [self.output]


class MinMaxOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML
        self.spark_convert_features = True

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()

        parameters = self.extract_field_value(parameters, "attribute")
        parameters = self.extract_field_value(parameters, "attributes")
        parameters = self.extract_field_value(parameters, "alias")

        if "attribute" in parameters:
            parameters['attributes'] = parameters['attribute']

        return parameters

    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows

        parameters = self.convert()
        self.features_col = parameters["attributes"]
        new_cols = parameters['alias'].split(",")

        for new_col, col in zip(new_cols, self.features_col):

            self.output.create_new_column(**{
                "name": new_col,
                "type": "DOUBLE",
                "missing_total": 0,
                'distinct_columns': self.input.columns[col].distinct_values,
                'min_value': 0,
                'max_value': 1
            })

        self.output.recalculate()

        return [self.output]

    def gen_model(self, platform_target=None):
        
        return {
            #v1
            "n_rows": self.input.n_rows,
            "n_features": len(self.features_col),
            "target_engine": self.platform_target
        }





class MLPClassifierOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, parameters):
        pass


class MLPRegressorOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, parameters):
        pass


class NaiveBayesClassificationOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, parameters):
        return parameters


class ProjectionOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        
        self.behavior = self.PHYSICAL_BEHAVIOR_SERIAL_LESS
        self.n_input = 1
        self.n_output = 1
        self.output = None
        self.input = None
        self.platform_id = self.get_platform_target()

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()

        if "value" in parameters.get("attributes", {}):
            parameters["attributes"] = parameters["attributes"]["value"]
        return parameters

    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows
        self.output = copy.deepcopy(base_statistics[0])
        
        parameters = self.convert()
        for col in list(self.output.columns.keys()):
            if col not in parameters["attributes"]:
                del self.output.columns[col]
#         self.output = BaseStatistics(n_rows=self.input.n_rows)

#         parameters = self.convert()
#         for col in parameters["attributes"]:
#             if col in self.input.columns:
#                 self.output.columns[col] = copy.deepcopy(self.input.columns[col])
        self.output.recalculate()
        # print(self.input)
        # print(self.output)
        return [self.output]

    def gen_model(self, platform_target=None):

        return {
            "input_n_rows": self.input.n_rows,
            #"input_size_bytes_memory": self.input.size_bytes_memory,
            #"output_size_bytes_memory": self.output.size_bytes_memory,
            "ratio_reduction_columns": self.output.n_columns / self.input.n_columns,
            "target_engine": self.platform_target
        }


class OneHotEncoderOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, parameters):
        return parameters



class OutlierDetectionOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class PCAOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class PerceptronClassifierOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class GenerateNGramsOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_SERIAL_EQ


class QuantileDiscretizerOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_WIDE_INFO_EQ


class RandomForestClassifierOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class RandomForestRegressorOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class RemoveDuplicatedRowsOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_HASH_REDUCE

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("attributes", {}):
            parameters["attributes"] = parameters["attributes"]["value"]
        return parameters

    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.output.size_bytes_memory
        self.total_input_rows = self.input.n_rows
        parameters = self.convert()

        if "attributes" in parameters:
            keys = parameters["attributes"]
        else:
            keys = list(self.output.columns.keys())
        
        self.n_subset = len(keys)
        distinct_values = []
        for k in keys:
            distinct_values.append(self.input.columns[k].distinct_values)
        
        def argmax(iterable):
            return max(enumerate(iterable), key=lambda x: x[1])[0]
        
        distinct_values_i = argmax(distinct_values)
        distinct_values_max = distinct_values[distinct_values_i]
        distinct_values_i = keys[distinct_values_i]
        
        
        if self.output.n_rows > distinct_values_max:
            to_remove = self.output.n_rows - distinct_values_max
            # print("distinct_values_max:", distinct_values_max)
            if self.debug:
                print(f"RemoveDuplicatedRowsOperationModel | distinct_values_max: {distinct_values_max} - to_remove: {to_remove}")
            self.output.remove_random_deciles(to_remove, except_list=[])
            self.output.remove_n_rows(to_remove) 
                    
            new_rows = self.output.n_rows

            for k, v in zip(keys, distinct_values):
                if v > new_rows:
                    v = new_rows
                self.output.columns[k].distinct_values = v

            self.output.recalculate()
        # print("In-remove:",self.input)
        # print("Out-remove:", self.output)
        # 
        if self.debug:
            print("RemoveDuplicatedRowsOperationModel | output: ", self.output)
            print("RemoveDuplicatedRowsOperationModel | ", self.output.columns)
            for c in self.output.columns:
                print(f"RemoveDuplicatedRowsOperationModel | deciles: {c}", self.output.columns[c].deciles)
        return [self.output]

    def gen_model(self, platform_target=None):

        return {
            "input_n_rows": self.input.n_rows,
            "output_n_rows": self.output.n_rows,
            "n_attributes": self.n_subset,
            "target_engine": self.platform_target
        }


class RegressorModelOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class RemoveStopWordsOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_SERIAL_EQ


class ReplaceValueOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_SERIAL_EQ
        self.n_input = 1
        self.n_output = 1
        self.output = None
        self.n_subset = None

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("attributes", {}):
            parameters["attributes"] = parameters["attributes"]["value"]
        return parameters


    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.output.size_bytes_memory
        self.total_input_rows = self.input.n_rows
        parameters = self.convert()

        if "attributes" in parameters:
            keys = parameters["attributes"]
        else:
            keys = list(self.output.columns.keys())

        self.n_subset = len(keys)
        # TODO: update histogram !!
        return [self.output]

    def gen_model(self, platform_target=None):

        return {
            "input_n_rows": self.input.n_rows,
            "output_n_rows": self.output.n_rows,
            #"output_size_bytes_memory": self.output.size_bytes_memory,
            "n_attributes": self.n_subset,
            "target_engine": self.platform_target
        }


class SampleOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_WIDE_INFO_LESS

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        
        # Temporary                  
        if isinstance(parameters.get("type"), dict):
            if isinstance(parameters["type"]["value"], dict):
                  parameters["type"] = parameters["type"]["value"]['value']
            else:
                  parameters["type"] = parameters["type"]["value"]

                  
        if isinstance(parameters.get("value"), dict):
            if isinstance(parameters["value"]["value"], dict):
                  parameters["value"] = parameters["value"]["value"]['value']
            else:
                  parameters["value"] = parameters["value"]["value"]
                  
        if "fraction" in parameters:
            if isinstance(parameters["fraction"], dict):
                 parameters["fraction"] = parameters["fraction"]["value"]
        return parameters


    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows
        self.output = copy.deepcopy(base_statistics[0])
        parameters = self.convert()
        
        type_mode = parameters["type"]
        
        if "head" == type_mode:
            n_rows = parameters["value"]
        #else:
        #    print("Not supported yet ")
            
        n_rows = int(parameters["value"])

        n_rows_to_remove = self.output.n_rows - n_rows

        if n_rows_to_remove > 0:
            self.output.remove_random_deciles(n_rows_to_remove, except_list=[])
            self.output.remove_n_rows(n_rows_to_remove)
            
            self.output.recalculate() 
            if self.debug:
                print("SampleOperationModel | output: ", self.output)
                for c in self.output.columns:
                    print("SampleOperationModel | input deciles: ", self.input.columns[c].deciles)
                    print("SampleOperationModel | output deciles: ", self.output.columns[c].deciles)
            
        return [self.output]

    def gen_model(self, platform_target=None):

        return {
            "input_n_rows": self.input.n_rows,
            "output_n_rows": self.output.n_rows,
            "reduction_ratio": self.input.n_rows / self.output.n_rows,
            "target_engine": self.platform_target
        }


class SetIntersectionOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_HASH_REDUCE


class SortOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_RANGE
        self.n_input = 1
        self.n_output = 1
        self.output = None

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("attributes", {}):
            parameters["attributes"] = parameters["attributes"]["value"]
        return parameters


    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.output.size_bytes_memory
        self.total_input_rows = self.input.n_rows
        return [self.output]

    def gen_model(self, platform_target=None):
        parameters = self.convert()
        keys = parameters["attributes"]

        return {
            "input_n_rows": self.input.n_rows,
            #"output_size_bytes_memory": self.output.size_bytes_memory,
            "n_columns": self.output.n_columns,
            "ratio_key": len(keys) / self.output.n_columns,
            "target_engine": self.platform_target
        }


class SplitOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_WIDE_INFO_EQ

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "weights" in parameters:
            if isinstance(parameters["weights"], dict):
                if "value" in parameters.get("weights", {}):
                    parameters["weights"] = float(parameters["weights"]["value"])
        return parameters



class SplitKFoldOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = "not-supported"


class SequenceMiningOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class SGBRegressorOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class StandardScalerOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML
        self.spark_convert_features = True

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()

        parameters = self.extract_field_value(parameters, "attributes")
        parameters = self.extract_field_value(parameters, "with_mean")
        parameters = self.extract_field_value(parameters, "with_std")
        parameters['attribute'] = parameters['attributes'] 



        return parameters

    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows

        parameters = self.convert()
        self.features_col = parameters["features"]
        self.label = parameters['label']
        n_distinct_values = self.input.columns[self.label].distinct_values
        
        self.output.create_new_column(**{
            "name": parameters["alias"],
            "type": "DOUBLE",
            "missing_total": 0,
            'distinct_columns': n_distinct_values,
            'min_value': 0,
            'max_value': n_distinct_values
        })
        self.output.recalculate()

        return [self.output]


class STDBSCANOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML


class SummaryStatisticsOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("alias", {}):
            parameters["alias"] = parameters["alias"]["value"]
        return parameters


class SVMClassificationOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_ML
        self.n_input = 1
        self.n_output = 1
        self.input = None
        self.output = None
        self.features_col = None

    def convert(self, platform_id=None):
        parameters = self.parameters.copy()

        if "value" in parameters.get("features", {}):
            parameters["features"] = parameters["features"]["value"]

        if "alias" in parameters:
            if "value" in parameters.get("alias", {}):
                parameters["alias"] = parameters["alias"]["value"]
        else:
            parameters["alias"] = "prediction"

        if "value" in parameters.get("max_iter", {}):
            parameters["max_iter"] = int(parameters["max_iter"]["value"])

        # to spark
        parameters["paramgrid"] = {"max_iter": parameters["max_iter"]}
        # to sklearn
        parameters["kernel"] = "linear"

        self.features['max_iter'] = int(parameters['max_iter'])
        return parameters



    def estimate_output(self, base_statistics):
        self.input = copy.deepcopy(base_statistics[0])
        self.output = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows

        parameters = self.convert()
        self.features_col = parameters["features"]
        # if len(col_features) > 0:
        #     self.output.col_as_features
        
        self.output.create_new_column(**{
            "name": parameters["alias"],
            "type": "DOUBLE",
            "missing_total": 0,
            'distinct_columns': 2,
            'min_value': 0,
            'max_value': 1
        })
        self.output.recalculate()

        return [self.output]
    
    def gen_model(self, platform_target=None):
        col1 = (self.input.n_rows**2) * len(self.features_col)**(1/5)
        col2 = ((self.features['max_iter']/100)**3) #.7182
        
        if platform_target == self.PLATFORM_PANDAS:
            return {
                #v1
                "n_rows": self.input.n_rows,
                "max_iter":  self.features['max_iter'],
                #"mix": (np.log(self.input.n_rows)  * col2)/100_000,
                "n_features": len(self.features_col), #**(1/5),
                "target_engine": self.platform_target
            }
        else:
            return {
                #v1
                 "n_rows": self.input.n_rows,
                "max_iter":  self.features['max_iter'],
                #"mix": (np.log(self.input.n_rows)  * col2)/100_000,
                "n_features": len(self.features_col),#**(1/5),
                "target_engine": self.platform_target
            }


class TransformationOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_SERIAL_EQ
        self.n_input = 1
        self.n_output = 1
        self.input = None
        self.output = None
        self.features_col = None


    def convert(self, platform_id=None):
        parameters = self.parameters.copy()
        if "value" in parameters.get("alias", {}):
            parameters["alias"] = parameters["alias"]["value"]

        if "value" in parameters.get("expression", {}):
            parameters["expression"] = parameters["expression"]["value"]
        return parameters



    def estimate_output(self, base_statistics):
        # Inicialmente assume que o tipo de saída da coluna será do mesmo tipo
        # da coluna de origem

        self.input = copy.deepcopy(base_statistics[0])
        self.total_input_size_bytes_memory = self.input.size_bytes_memory
        self.total_input_rows = self.input.n_rows
        self.output = copy.deepcopy(base_statistics[0])
        parameters = self.convert()

        self.n_functions = len(parameters["expression"])
        self.overwrite = 0
        self.n_columns = 0 
        
        for exp in parameters["expression"]:
            new_col = exp["alias"]
            self.n_columns += (json.dumps(exp).count("Identifier")+1)**2 * self.input.n_rows
            if new_col in self.output.columns:
                self.overwrite += 1
            
            col_name = None
            if "name" in exp["tree"]:
                col_name = exp["tree"]["name"]
            elif "name" in exp["tree"].get("left", []):
                col_name = exp["tree"]["left"]["name"]
            elif "name" in exp["tree"].get("right", []):
                col_name = exp["tree"]["right"]["name"]
            elif "name" in exp["tree"].get("arguments", [[]])[0]:
                col_name = exp["tree"].get("arguments", [[]])[0]["name"]
            
            if col_name:
                col_object = copy.deepcopy(self.output.columns[col_name])
                col_object.name = new_col
                self.output.columns[new_col] = col_object
            else:
                # literal
                # todo: ler o literal e pegar o valor exato
                self.output.create_new_column(**{
                    "n_rows": self.output.n_rows,
                    "name": new_col,
                    "type": "INTEGER",
                    "missing_total": 0,
                    'min_value': 1,
                    'max_value': 1,
                    'distinct_values': 1,

                })

            

        self.output.recalculate()

        return [self.output]

    def gen_model(self, platform_target=None):

        return {
            #"output_bytes_per_row": self.output.size_bytes_memory / self.output.n_rows, 
            "n_rows": self.input.n_rows,
            "n_rows_plus_columns": self.n_columns,
            'n_overwrite_column': self.overwrite,
            #"n_used_columns": self.n_columns,
            "target_engine": self.platform_target
        }


class TokenizerOperationModel(OperationModel):
    def __init__(self, parameters):
        OperationModel.__init__(self, parameters)
        self.features = {}
        self.behavior = self.PHYSICAL_BEHAVIOR_SERIAL_EQ

    def convert(self, parameters):
        return parameters



class VisualizationOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        
        self.behavior = self.PHYSICAL_BEHAVIOR_VISUALIZATION


class WordToVectorOperationModel(GenericOperationModel):
    def __init__(self, parameters):
        GenericOperationModel.__init__(self, parameters)
        self.behavior = self.PHYSICAL_BEHAVIOR_ML
