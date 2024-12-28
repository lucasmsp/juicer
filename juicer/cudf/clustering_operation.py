# -*- coding: utf-8 -*-

from textwrap import dedent
from juicer.operation import Operation
from juicer.scikit_learn.util import get_X_train_data
from juicer.scikit_learn.model_operation import AlgorithmOperation
from gettext import gettext

class ClusteringOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs, algorithm):
        input_data = named_inputs.get('train input data')
        if input_data is None:
            input_data = named_inputs.get('input data')
        model_in_ports = {
            'train input data': input_data,
            'algorithm': 'algorithm'}
        model = ClusteringModelOperation(
            parameters, model_in_ports, named_outputs)
        super(ClusteringOperation, self).__init__(
            parameters, named_inputs, named_outputs, model, algorithm)
        model.metrics_code = algorithm.get_output_metrics_code()


class ClusteringModelOperation(Operation):
    FEATURES_PARAM = 'features'
    ALIAS_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) >= 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])

        if self.has_code:
            if self.FEATURES_PARAM in parameters:
                self.features = parameters.get(self.FEATURES_PARAM)
            else:
                raise \
                    ValueError(_("Parameter '{}' must be informed for task {}")
                               .format(self.FEATURES_PARAM, self.__class__))

            self.model = self.named_outputs.get(
                'model', 'model_task_{}'.format(self.order))

            self.output = self.named_outputs.get(
                'output data', 'out_task_{}'.format(self.order))
            self.alias = parameters.get(self.ALIAS_PARAM, 'prediction')

            self.transpiler_utils.add_custom_function('get_X_train_data',
                                                      get_X_train_data)
            self.metrics_code = ""

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        if self.has_code:
            """Generate code."""

            copy_code = ".copy()" \
                if self.parameters['multiplicity'].get(
                'train input data', 0) > 1 or \
                   self.parameters['multiplicity'].get(
                       'input data', 0) > 1 \
                else ""

            code = """
        for c in {features}:
            {input}[c] = {input}[c].astype('float32')
        
        X = {input}[{features}]
        clustering_model = {algorithm}.fit(X)

        emit_event(name='update task', message=_('Model fitted.'),
               identifier=task_id, status='RUNNING')

        if hasattr(clustering_model, 'labels_'):
            y = clustering_model.labels_
        elif hasattr(clustering_model, 'predict'):
            y = clustering_model.predict(X)
        else:
            y = clustering_model.transform(X).tolist()

        {OUT} = {input}{copy_code}
        {OUT}['{predCol}'] = y
        {model} = clustering_model
        display_text = {display_text}
        if display_text:
            metric_rows = [{metrics_append}]

            if metric_rows:
                metrics_content = SimpleTableReport(
                    'table table-striped table-bordered w-auto', [],
                    metric_rows,
                    title='{metrics}')

                emit_event('update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=metrics_content.generate(),
                    type='HTML', title='{metrics}',
                    task={{'id': '{task_id}' }},
                    operation={{'id': {operation_id} }},
                    operation_id={operation_id})
            """.format(model=self.model, features=self.features,
                       input=self.named_inputs['train input data'],
                       algorithm=self.named_inputs['algorithm'],
                       copy_code=copy_code, OUT=self.output,
                       predCol=self.alias,
                       metrics_append=self.metrics_code,
                       task_id=self.parameters['task_id'],
                       operation_id=self.parameters['operation_id'],
                       title=_("Clustering result"),
                       summary=gettext('Summary'),
                       metrics=gettext('Metrics'),
                       weights=gettext('Weights'),
                       compute_cost=gettext('Compute cost'),

                       msg1=_('Regression only support numerical features.'),
                       msg2=_('Features are not assembled as a vector. '
                              'They will be implicitly assembled and rows with '
                              'null values will be discarded. If this is '
                              'undesirable, explicitly add a feature assembler '
                              'in the workflow.'),
                       display_text=self.parameters['task']['forms'].get(
                           'display_text', {}).get('value') in (1, '1')
                       )

            return dedent(code)

class KMeansModelOperation(ClusteringOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = KMeansClusteringOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(KMeansModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)

class KMeansClusteringOperation(Operation):
    N_CLUSTERS_PARAM = 'n_clusters'
    MAX_ITER_PARAM = 'max_iter'
    INIT_PARAM = 'init'

    TOLERANCE_PARAM = 'tol'
    SEED_PARAM = 'random_state'
    N_INIT_PARAM = 'n_init'


    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'

    INIT_PARAM_RANDOM = 'random'
    INIT_PARAM_KM = 'k-means||'


    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.has_code:
            self.n_init = int(parameters.get(self.N_INIT_PARAM, 10) or 10)
            self.tol = float(parameters.get(self.TOLERANCE_PARAM, 1e-4) or 1e-4)

            self.n_clusters = int(parameters.get(self.N_CLUSTERS_PARAM, 8) or 8)
            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, 100) or 100)
            self.init_mode = parameters.get(
                self.INIT_PARAM, self.INIT_PARAM_KM) or self.INIT_PARAM_KM
            self.init_mode = self.init_mode.lower()
            self.tolerance = parameters.get(self.TOLERANCE_PARAM, 1e-4)
            self.tolerance = abs(float(self.tolerance))
            self.seed = parameters.get(self.SEED_PARAM, None) or None

            vals = [self.n_clusters, self.max_iter]
            atts = [self.N_CLUSTERS_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                            att, self.__class__))


            self.transpiler_utils.add_custom_function(
                'get_X_train_data', get_X_train_data)
            #self.transpiler_utils. \
            #    add_import("from sklearn.metrics import silhouette_score")
            self.input_treatment()

    def input_treatment(self):
        pass


    @staticmethod
    def get_output_metrics_code():
        return ""
        # code = """
        #     #['{silhouette_euclidean}', silhouette_score(X, y, metric='euclidean')],
        #     #['{silhouette_cosine}', silhouette_score(X, y, metric='cosine')],
        #     ['{cluster_centers}', clustering_model.cluster_centers_],
        #     ['{inertia}', clustering_model.inertia_]
        #     """.format(inertia=gettext('Inertia'),
        #                cluster_centers=gettext('Cluster centers'),
        #                silhouette_euclidean=
        #                               gettext('Silhouette (Euclidean distance)'),
        #                silhouette_cosine=
        #                               gettext('Silhouette (Cosine distance)'))
        # return code

    def generate_code(self):
        """Generate code."""
        if self.seed:
            random_state = "random_state={seed}".format(seed=self.seed)
        else:
            random_state = ""

        code = """
            import cuml.cluster as cuml_cluster

            algorithm = cuml_cluster.KMeans(n_clusters={k}, init='{init}', 
                               max_iter={max_iter}, tol={tol}, 
                               {random_state} n_init={n_init})
            """.format(k=self.n_clusters, max_iter=self.max_iter,
                       tol=self.tolerance, init=self.init_mode, random_state=random_state,
                       n_init=self.n_init)

        return dedent(code)
