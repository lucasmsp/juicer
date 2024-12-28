# -*- coding: utf-8 -*-

from textwrap import dedent
import re
from juicer.operation import Operation
from juicer.util.template_util import *
from juicer.scikit_learn.util import get_X_train_data, get_label_data
from juicer.scikit_learn.model_operation import AlgorithmOperation

class ClassificationModelOperation(Operation):

    LABEL_ATTRIBUTE_PARAM = 'label'
    FEATURES_ATTRIBUTE_PARAM = 'features'
    PREDICTION_ATTRIBUTE_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) >= 1 and any(
                [len(self.named_outputs) >= 1, self.contains_results()])

        if not self.has_code and len(self.named_outputs) > 0:
            raise ValueError(
                    _('Model is being used, but at least one input is missing'))

        if any([self.FEATURES_ATTRIBUTE_PARAM not in parameters,
                self.LABEL_ATTRIBUTE_PARAM not in parameters]):
            msg = _("Parameters '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format(
                    self.FEATURES_ATTRIBUTE_PARAM, self.LABEL_ATTRIBUTE_PARAM,
                    self.__class__.__name__))

        self.label = parameters.get(self.LABEL_ATTRIBUTE_PARAM)
        self.features = parameters.get(self.FEATURES_ATTRIBUTE_PARAM)
        self.prediction = parameters.get(self.PREDICTION_ATTRIBUTE_PARAM,
                                         'prediction')

        self.model = self.named_outputs.get('model',
                                            'model_task_{}'.format(self.order))
        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.metrics_code = ""
        self.transpiler_utils.add_custom_function(
                'get_X_train_data', get_X_train_data)
        self.transpiler_utils.add_custom_function(
                'get_label_data', get_label_data)

        self.perform_cross_validation = parameters.get(
                'apply_cross_validation') in [True, '1', 1]
        if self.perform_cross_validation:
            self.transpiler_utils.add_import("from sklearn.model_selection "
                                             "import cross_val_score, KFold")
            self.cross_validation_metric = \
                parameters.get('metric_cross_validation', 'f1_weighted')
            self.kfold = int(parameters.get('folds', 3))

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        """Generate code."""
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['train input data'] > 1 else ""
        if self.perform_cross_validation:
            fit_code = "avg_score = cross_val_score(algorithm, X, y, " \
                       "cv={folds}, scoring='{metric}').mean()" \
                .format(folds=self.kfold, seed=None,
                        metric=self.cross_validation_metric)
            score = '["Average score in cross-validation ({k}-fold)", ' \
                    'avg_score],'.format(k=self.kfold)
        else:
            score = ""
            fit_code = ""

        code = """
        for c in {features}:
            {input}[c] = {input}[c].astype('float32')
        
        X = {input}[{features}]
        y = {input}[{label}]
        classification_model = algorithm.fit(X, y)
        {fit_code}
        {output} = {input}{copy_code}
        prediction = classification_model.predict(X)
        {output}['{predCol}'] = prediction
        {model} = classification_model

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
            """.format(model=self.model, label=self.label, copy_code=copy_code,
                       input=self.named_inputs['train input data'],
                       score=score,
                       features=self.features, output=self.output,
                       fit_code=fit_code,
                       predCol=self.prediction,
                       metrics_append=self.metrics_code,
                       task_id=self.parameters['task_id'],
                       operation_id=self.parameters['operation_id'],
                       title=_("Clustering result"),
                       summary=gettext('Summary'),
                       metrics=gettext('Metrics'),
                       display_text=self.parameters['task']['forms'].get(
                               'display_text', {}).get('value') in (1, '1'))

        return dedent(code)


class ClassificationOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs, algorithm):
        model_in_ports = {
            'train input data': named_inputs.get('train input data'),
            'algorithm': 'algorithm'}

        model = ClassificationModelOperation(
                parameters, model_in_ports, named_outputs)
        super(ClassificationOperation, self).__init__(
                parameters, named_inputs, named_outputs, model, algorithm)
        model.metrics_code = algorithm.get_output_metrics_code()

class LogisticRegressionModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = LogisticRegressionOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(LogisticRegressionModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class LogisticRegressionOperation(Operation):
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'
    TOLERANCE_PARAM = 'tol'
    PENALTY_PARAM = 'penalty' # ‘none’, ‘l1’, ‘l2’, ‘elasticnet’ (default = ‘l2’)
    MAX_ITER_PARAM = 'max_iter'
    SOLVER_PARAM = 'solver'
    REGULARIZATION_PARAM = 'C'
    FIT_INTERCEPT_PARAM = 'fit_intercept'


    SOLVER_PARAM_QN =  'qn'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.has_code:
            if self.LABEL_PARAM not in parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.LABEL_PARAM,
                    self.__class__))
            else:
                self.label = parameters.get(self.LABEL_PARAM, None)

            if self.FEATURES_PARAM not in parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM,
                    self.__class__))
            else:
                self.features = parameters.get(self.FEATURES_PARAM, None)

            self.prediction_column = parameters.get(
                    self.PREDICTION_PARAM, 'prediction')

            self.tol = float(self.parameters.get(
                    self.TOLERANCE_PARAM, 0.0001) or 0.0001)
            if self.tol <= 0:
                raise ValueError(
                    _("Parameter '{}' must be x>0 for task {}").format(
                        self.TOLERANCE_PARAM, self.__class__))

            self.regularization = float(self.parameters.get(
                self.REGULARIZATION_PARAM, 1.0)) or 1.0
            self.max_iter = int(self.parameters.get(
                    self.MAX_ITER_PARAM, 100)) or 100

            self.solver = self.parameters.get(
                    self.SOLVER_PARAM, self.SOLVER_PARAM_QN)\
                or self.SOLVER_PARAM_QN

            self.penalty = parameters.get(self.PENALTY_PARAM, 'l2')
            self.fit_intercept = int(parameters.get(
                    self.FIT_INTERCEPT_PARAM, 1)) == 1


            vals = [self.regularization, self.max_iter]
            atts = [self.REGULARIZATION_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            solver_dict = {
                'qn': ['l2', 'none', 'l1', 'elasticnet']
            }

            self.transpiler_utils.add_import(
                    "import cuml.linear_model as cuml_linear_model")

    @staticmethod
    def get_output_metrics_code():
        code = """
        ['{coef}', classification_model.coef_],
        ['{intercept}', classification_model.intercept_],
        """.format(coef=gettext('Coefficient'),
                   intercept=gettext('Intercept'))
        return code

    def generate_code(self):
        """Generate code."""
        code = """
        algorithm = cuml_linear_model.LogisticRegression(tol={tol}, C={C}, 
            max_iter={max_iter}, solver='qn', 
            penalty='{penalty}',
            fit_intercept={fit_intercept})
        """.format(tol=self.tol, C=self.regularization,
                   max_iter=self.max_iter,
                   solver=self.solver, penalty=self.penalty,
                   fit_intercept=self.fit_intercept)
        return code


class SvmClassifierModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = SvmClassifierOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(SvmClassifierModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)

class SvmClassifierOperation(Operation):
    PENALTY_PARAM = 'c'
    KERNEL_PARAM = 'kernel'
    DEGREE_PARAM = 'degree'
    TOLERANCE_PARAM = 'tol'
    MAX_ITER_PARAM = 'max_iter'
    SEED_PARAM = 'seed'
    LABEL_PARAM = 'label'
    FEATURES_PARAM = 'features'
    PREDICTION_PARAM = 'prediction'
    GAMMA_PARAM = 'gamma'
    COEF0_PARAM = 'coef0'
    PROBABILITY_PARAM = 'probability'


    KERNEL_PARAM_LINEAR = 'linear'
    KERNEL_PARAM_RBF = 'rbf'
    KERNEL_PARAM_POLY = 'poly'
    KERNEL_PARAM_SIG = 'sigmoid'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.has_code:

            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, -1))
            self.tol = float(parameters.get(self.TOLERANCE_PARAM, 0.001) or
                             0.001)
            self.tol = abs(float(self.tol))
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.degree = int(parameters.get(self.DEGREE_PARAM, 3) or 3)
            self.kernel = parameters.get(
                    self.KERNEL_PARAM,
                    self.KERNEL_PARAM_RBF) or self.KERNEL_PARAM_RBF
            self.c = float(parameters.get(self.PENALTY_PARAM, 1.0) or 1.0)

            self.gamma = parameters.get(self.GAMMA_PARAM, 'scale')

            self.coef0 = float(parameters.get(self.COEF0_PARAM, 0.0) or 0.0)

            self.probability = int(parameters.get(
                    self.PROBABILITY_PARAM, 0)) == 1


            vals = [self.degree, self.c]
            atts = [self.DEGREE_PARAM, self.PENALTY_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.transpiler_utils.add_import("from sklearn.svm import SVC")

    def get_output_metrics_code(self):
        code = """
        ['{intercept}', classification_model.intercept_],
        ['{classes}', classification_model.classes_],
        """.format(intercept=gettext('Intercept'),
                   classes=gettext('The classes labels'),
                   class_weight=gettext('Multipliers of parameter C '
                                        'for each class'))
        if self.kernel == self.KERNEL_PARAM_LINEAR:
            code += """
            ['{coef}', classification_model.coef_],
            """.format(coef=gettext('Coefficient'))
        return code

    def generate_code(self):
        """Generate code."""

        code = """
        import cuml.svm as cuml_svm
        
        algorithm = cuml_svm.SVC(tol={tol}, C={c}, max_iter={max_iter}, 
                       degree={degree}, kernel='{kernel}', random_state={seed},
                       gamma='{gamma}', coef0={coef0}, probability={prob},
                       class_weight=None)
        """.format(tol=self.tol, c=self.c, max_iter=self.max_iter,
                   degree=self.degree, kernel=self.kernel, seed=self.seed,
                   gamma=self.gamma, coef0=self.coef0, prob=self.probability)
        return code
