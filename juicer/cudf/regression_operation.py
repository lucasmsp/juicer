# -*- coding: utf-8 -*-

from textwrap import dedent
from juicer.operation import Operation
import re
from juicer.scikit_learn.util import get_X_train_data, get_label_data
from juicer.util.template_util import *
from juicer.scikit_learn.model_operation import AlgorithmOperation


class RegressionModelOperation(Operation):
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_COL_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) >= 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])

        if self.has_code:
            self.algorithm = self.named_inputs['algorithm']
            self.input = self.named_inputs['train input data']

            if not all([self.FEATURES_PARAM in parameters,
                        self.LABEL_PARAM in parameters]):
                msg = _("Parameters '{}' and '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM, self.LABEL_PARAM,
                    self.__class__.__name__))

            self.features = parameters[self.FEATURES_PARAM]
            self.label = parameters[self.LABEL_PARAM]
            self.prediction = parameters.get(self.PREDICTION_COL_PARAM,
                                             'prediction') or 'prediction'
            self.model = self.named_outputs.get(
                'model', 'model_task_{}'.format(self.order))
            self.output = self.named_outputs.get(
                'output data', 'out_task_{}'.format(self.order))
            self.transpiler_utils.add_custom_function(
                'get_X_train_data', get_X_train_data)
            self.transpiler_utils.add_custom_function(
                'get_label_data', get_label_data)
            self.metrics_code = ""

            self.perform_cross_validation = parameters.get(
                'apply_cross_validation') in [True, '1', 1]
            if self.perform_cross_validation:
                self.transpiler_utils.add_import("from sklearn.model_selection "
                                                 "import cross_val_score, "
                                                 "KFold")
                self.cross_validation_metric = \
                    parameters.get('metric_cross_validation', 'r2')
                self.kfold = int(parameters.get('folds', 3))

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        copy_code = ".copy()" \
            if self.parameters['multiplicity'].get('train input data', 0) > 1 \
               or self.parameters['multiplicity'].get('input data', 0) > 1 \
            else ""

        if self.perform_cross_validation:
            fit_code = "avg_score = cross_val_score(algorithm, X_train, y, " \
                       "cv={folds}, scoring='{metric}').mean()" \
                .format(folds=self.kfold, seed=None,
                        metric=self.cross_validation_metric)
            score = '["Average score in cross-validation ({k}-fold)", ' \
                    'avg_score],'.format(k=self.kfold)
        else:
            score = ""
            fit_code = ""

        code = """
        X_train = {input}[{features}]
        if 'IsotonicRegression' in str(algorithm):
            X_train = np.ravel(X_train)
        y = {input}{label}
        regressor_model = algorithm.fit(X_train, y, convert_dtype=True)
        {fit_code}
        {output_data} = {input}{copy_code}
        prediction = algorithm.predict(X_train)
        {output_data}['{prediction}'] = prediction
        {model} = regressor_model

        display_text = {display_text}
        if display_text:
            metric_rows = [{score}{metrics_append}]

            if metric_rows:
                metrics_content = SimpleTableReport(
                    'table table-striped table-bordered w-auto', [],
                    metric_rows, title='{metrics}')

                emit_event('update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=metrics_content.generate(),
                    type='HTML', title='{metrics}',
                    task={{'id': '{task_id}' }},
                    operation={{'id': {operation_id} }},
                    operation_id={operation_id})
        """.format(copy_code=copy_code, model=self.model,
                   algorithm=self.algorithm,
                   score=score, fit_code=fit_code,
                   input=self.named_inputs['train input data'],
                   output_data=self.output, prediction=self.prediction,
                   label=self.label, features=self.features,
                   metrics_append=self.metrics_code,
                   task_id=self.parameters['task_id'],
                   operation_id=self.parameters['operation_id'],
                   title=_("Clustering result"),
                   summary=gettext('Summary'),
                   metrics=gettext('Metrics'),
                   display_text=self.parameters['task']['forms'].get(
                       'display_text', {}).get('value') in (1, '1'))

        return dedent(code)


class RegressorOperationOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs, algorithm):
        input_data = named_inputs.get('train input data')
        if input_data is None:
            input_data = named_inputs.get('input data')
        model_in_ports = {
            'train input data': input_data,
            'algorithm': 'algorithm'}

        model = RegressionModelOperation(
            parameters, model_in_ports, named_outputs)
        super(RegressorOperationOperation, self).__init__(
            parameters, named_inputs, named_outputs, model, algorithm)
        model.metrics_code = algorithm.get_output_metrics_code()



class RandomForestRegressorModelOperation(RegressorOperationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = RandomForestRegressorOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(RandomForestRegressorModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)

class RandomForestRegressorOperation(Operation):
    # cuml.ensemble.RandomForestRegressor
    """
    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    """

    N_ESTIMATORS_PARAM = 'n_estimators'
    CRITERION_PARAM = 'stop_criterion'
    BOOTSTRAP_PARAM = 'bootstrap'
    MAX_DEPTH_PARAM = 'max_depth'
    MAX_LEAF_NODES_PARAM = 'max_leaves'
    MAX_FEATURES_PARAM = 'max_features'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    RANDOM_STATE_PARAM = 'random_state'

    PREDICTION_PARAM = 'prediction'
    LABEL_PARAM = 'label'
    FEATURES_PARAM = 'features'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.parameters = parameters
        self.name = 'curfr.RandomForestRegressor'
        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])

        if self.has_code:
            self.n_estimators = int(
                parameters.get(self.N_ESTIMATORS_PARAM, 100) or 100)
            self.max_features = parameters.get(self.MAX_FEATURES_PARAM,
                                               'auto') or 'auto'
            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM, 16)
            self.min_samples_split = int(
                parameters.get(self.MIN_SPLIT_PARAM, 2) or 2)
            self.min_samples_leaf = int(
                parameters.get(self.MIN_LEAF_PARAM, 1) or 1)
            self.criterion = parameters.get(self.CRITERION_PARAM,
                                            'mse') or 'mse'

            self.max_leaf_nodes = parameters.get(self.MAX_LEAF_NODES_PARAM,
                                                 -1)
            self.bootstrap = int(parameters.get(self.BOOTSTRAP_PARAM, True))
            self.random_state = parameters.get(self.RANDOM_STATE_PARAM, None)

            vals = [self.n_estimators, self.min_samples_split,
                    self.min_samples_leaf]
            atts = [self.N_ESTIMATORS_PARAM, self.MIN_SPLIT_PARAM,
                    self.MIN_LEAF_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.input_treatment()


    def input_treatment(self):

        if self.max_depth is not None:
            self.max_depth = int(self.max_depth)
            if self.max_depth <= 0:
                raise ValueError(
                    _("Parameter '{}' must be x>0 or None for task {}").format(
                        self.MAX_DEPTH_PARAM, self.__class__))
        else:
            self.max_depth = None

        if self.max_leaf_nodes is not None and self.max_leaf_nodes != '0':
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        else:
            self.max_leaf_nodes = None


    def get_output_metrics_code(self):
        if self.has_code:
            code = """
            """
            # ['{feature_importances}', regressor_model.feature_importances_],
            # ['{n_features}', regressor_model.n_features_],
            # ['{n_outputs}', regressor_model.n_outputs_],
            # """.format(feature_importances=gettext('The impurity-based '
            #                                        'feature importances'),
            #            n_features=gettext('The number of features'),
            #            n_outputs=gettext('The number of outputs'))

            return code

    def generate_code(self):
        if self.has_code:
            code = """
            import cuml.ensemble as curfr
            
            algorithm = curfr.RandomForestRegressor(n_estimators={n_estimators}, 
                max_features='{max_features}', 
                max_depth={max_depth}, 
                min_samples_split={min_samples_split}, 
                min_samples_leaf={min_samples_leaf}, 
                split_criterion='{stop_criterion}',
                max_leaves={max_leaves}, 
                bootstrap={bootstrap})
            """.format(n_estimators=self.n_estimators,
                       max_features=self.max_features,
                       max_depth=self.max_depth,
                       min_samples_split=self.min_samples_split,
                       min_samples_leaf=self.min_samples_leaf,
                       stop_criterion=self.criterion,
                       max_leaves=self.max_leaf_nodes,
                       bootstrap=self.bootstrap)

            return code

class LinearRegressionModelOperation(RegressorOperationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = LinearRegressionOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(LinearRegressionModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


# TypeError: Expected input to be of type in [dtype('float32'), dtype('float64')] but got int64
class LinearRegressionOperation(Operation):

    ALPHA_PARAM = 'alpha'
    ELASTIC_NET_PARAM = 'l1_ratio'
    NORMALIZE_PARAM = 'normalize'
    FIT_INTERCEPT_PARAM = 'fit_intercept'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tol'

    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'

    # ‘cd’, ‘qn’

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.name = 'regression.LinearRegression'
        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])

        if self.has_code:
            self.alpha = float(parameters.get(self.ALPHA_PARAM, 1.0) or 1.0)
            self.elastic = float(
                parameters.get(self.ELASTIC_NET_PARAM, 0.5) or 0.5)

            self.normalize = False
            #self.normalize = self.parameters.get(self.NORMALIZE_PARAM, True) in (1, '1', 'true', True)
            self.max_iter = int(
                parameters.get(self.MAX_ITER_PARAM, 1000) or 1000)
            self.tol = float(
                self.parameters.get(self.TOLERANCE_PARAM, 0.0001) or 0.0001)

            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM,
                                                  'prediction')

            self.fit_intercept = self.parameters.get(self.FIT_INTERCEPT_PARAM,
                                                     False) == 1


            vals = [self.alpha, self.max_iter]
            atts = [self.ALPHA_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            if self.elastic < 0 or self.elastic > 1:
                raise ValueError(
                        _("Parameter '{}' must be 0<=x<=1 for task {}").format(
                                self.ELASTIC_NET_PARAM, self.__class__))


    @staticmethod
    def get_output_metrics_code():
        code = """
        ['{coef}', regressor_model.coef_],
        ['{intercept}', regressor_model.intercept_],
        """.format(coef=gettext("Parameter vector"),
                   intercept=gettext("Intercept"))
        return code

    def generate_code(self):
        if self.has_code:
            code = """    
            import cuml.linear_model as cuml_linear_model
            algorithm = cuml_linear_model.ElasticNet(alpha={alpha}, l1_ratio={elastic}, tol={tol}, 
                max_iter={max_iter}, normalize={normalize}, fit_intercept={fit_intercept})  
            """.format(max_iter=self.max_iter,
                       alpha=self.alpha,
                       elastic=self.elastic,
                       tol=self.tol,
                       normalize=self.normalize,
                       fit_intercept=self.fit_intercept)

            return code
