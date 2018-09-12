# coding=utf-8
from __future__ import absolute_import

import hashlib
import sys
import uuid
from collections import OrderedDict

import datetime

import jinja2
import networkx as nx
from .service import stand_service
from .util.jinja2_custom import AutoPep8Extension
from .util.template_util import HandleExceptionExtension


class DependencyController(object):
    """ Evaluates if a dependency is met when generating code. """

    def __init__(self, requires):
        self._satisfied = set()
        self.requires = requires

    def satisfied(self, _id):
        self._satisfied.add(_id)

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def is_satisfied(self, _id):
        return True  # len(self.requires[_id].difference(self._satisfied)) == 0


# noinspection PyMethodMayBeStatic
class Transpiler(object):
    """Base class for transpilers (converts workflow into platform specific
    code).
    """
    VISITORS = []
    DATA_SOURCE_OPS = ['data-reader']
    __slots__ = (
        'configuration', 'current_task_id', 'operations', 'port_id_to_port',
        'slug_to_op_id', 'template_dir'
    )

    def __init__(self, configuration, template_dir, slug_to_op_id=None,
                 port_id_to_port=None):
        self.operations = {}
        if slug_to_op_id is None:
            self.slug_to_op_id = {}
        else:
            self.slug_to_op_id = slug_to_op_id
        if port_id_to_port is None:
            self.port_id_to_port = {}
        else:
            self.port_id_to_port = port_id_to_port
        self.operations = {}
        self._assign_operations()
        self.configuration = configuration
        self.template_dir = template_dir
        self.current_task_id = None

    def _assign_operations(self):
        raise NotImplementedError()

    def get_context(self):
        return {}

    def get_code_template(self):
        return "templates/operation.tmpl"

    def get_notebook_template(self):
        return "templates/notebook.tmpl"

    def get_deploy_template(self):
        return "templates/deploy.tmpl"

    def generate_code(self, graph, job_id, out, params, ports,
                      sorted_tasks_id, state, task_hash, using_stdout,
                      workflow, deploy=False, export_notebook=False):

        if deploy:
            # To be able to convert, workflow must obey all these rules:
            # - 1 and exactly 1 data source;
            # - Data source must be defined in Limonero with its attributes in
            # order to define the schema for data input;
            # - For ML models, it is required to have a Save Model operation;
            total_ds = 0
            for task in workflow['tasks']:
                if not task.get('enabled', False):
                    continue
                if task['operation']['slug'] in self.DATA_SOURCE_OPS:
                    total_ds += 1

            if total_ds < 1:
                raise ValueError(_(
                    'Workflow must have at least 1 data source to be deployed.')
                )
            tasks_ids = reversed(sorted_tasks_id)
        else:
            tasks_ids = sorted_tasks_id

        instances = OrderedDict()
        for i, task_id in enumerate(tasks_ids):
            task = graph.node[task_id]
            self.current_task_id = task_id
            class_name = self.operations[task['operation']['slug']]

            parameters = {}
            not_empty_params = [(k, d) for k, d in task['forms'].items() if
                                d['value']]

            task['forms'] = dict(not_empty_params)
            for parameter, definition in task['forms'].items():
                # @FIXME: Fix wrong name of form category
                # (using name instead of category)
                cat = definition.get('category',
                                     'execution').lower()  # FIXME!!!
                cat = 'paramgrid' if cat == 'param grid' else cat
                cat = 'logging' if cat == 'execution logging' else cat

                if all([cat in ["execution", 'paramgrid', 'param grid',
                                'execution logging', 'logging'],
                        definition['value'] is not None]):

                    task_hash.update(unicode(definition['value']).encode(
                        'utf8', errors='ignore'))
                    if cat in ['paramgrid', 'logging']:
                        if cat not in parameters:
                            parameters[cat] = {}
                        parameters[cat][parameter] = definition['value']
                    else:
                        parameters[parameter] = definition['value']
                # escape invalid characters for code generation
                # except JSON (starting with {)
                if definition['value'] is not None and not isinstance(
                        definition['value'], bool):
                    if '"' in definition['value'] or "'" in definition['value']:
                        if definition['value'][0] != '{':
                            definition['value'] = TranspilerUtils.escape_chars(
                                definition['value'])

            if state is None or state.get(task_id) is None:
                parameters['execution_date'] = None
            else:
                v = state.get(task_id, [{}])[0]
                if v:
                    parameters['execution_date'] = v.get('execution_date')
                else:
                    parameters['execution_date'] = None
            true_values = (1, '1', True, 'true', 'True')
            parameters.update({
                'configuration': self.configuration,
                'display_sample': task['forms'].get('display_sample', {}).get(
                    'value') in true_values,
                'display_schema': task['forms'].get('display_schema', {}).get(
                    'value') in true_values,
                # Hash is used in order to avoid re-run task.
                'hash': task_hash.hexdigest(),
                'job_id': job_id,
                'operation_id': task['operation']['id'],
                'operation_slug': task['operation']['slug'],
                # Some temporary variables need to be identified by a sequential
                # number, so it will be stored in this field
                'order': i,
                'task': task,
                'task_id': task['id'],
                'user': workflow['user'],
                'workflow': workflow,
                'workflow_id': workflow['id'],
                # Some operations require the complete workflow data
                'workflow_name': TranspilerUtils.escape_chars(workflow['name']),
                'export_notebook': export_notebook,
            })
            port = ports.get(task['id'], {})

            instance = class_name(parameters, port.get('named_inputs', {}),
                                  port.get('named_outputs', {}))
            instance.out_degree = graph.out_degree(task_id)

            instances[task['id']] = instance

        env_setup = {
            'dependency_controller': DependencyController(
                params.get('requires_info', False)),
            'disabled_tasks': workflow['disabled_tasks'],
            'execute_main': params.get('execute_main', False),
            'instances': list(instances.values()),
            'instances_by_task_id': instances,
            'now': datetime.datetime.now(), 'user': workflow['user'],
            'plain': params.get('plain', False),
            'transpiler': TranspilerUtils(),
            'workflow_name': workflow['name'],
        }
        env_setup.update(self.get_context())

        template_loader = jinja2.FileSystemLoader(
            searchpath=self.template_dir)
        template_env = jinja2.Environment(loader=template_loader,
                                          extensions=[AutoPep8Extension,
                                                      HandleExceptionExtension,
                                                      'jinja2.ext.do'])
        template_env.globals.update(zip=zip)

        if deploy:
            env_setup['slug_to_op_id'] = self.slug_to_op_id
            # env_setup['slug_to_port_id'] = self.slug_to_port_id
            env_setup['id_mapping'] = {}
            template = template_env.get_template(self.get_deploy_template())
            v = template.render(env_setup)
            out.write(v.encode('utf8'))
        elif export_notebook:
            template = template_env.get_template(self.get_notebook_template())
            v = template.render(env_setup)
            out.write(v.encode('utf8'))
        else:
            template = template_env.get_template(self.get_code_template())
            v = template.render(env_setup)
            if using_stdout:
                out.write(v.encode('utf8'))
            else:
                out.write(v)
            stand_config = self.configuration.get('juicer', {}).get(
                'services', {}).get('stand')
            if stand_config and job_id:
                # noinspection PyBroadException
                try:
                    stand_service.save_job_source_code(
                        stand_config['url'], stand_config['auth_token'], job_id,
                        v.encode('utf8'))
                except:
                    pass

    def transpile(self, workflow, graph, params, out=None, job_id=None,
                  state=None, deploy=False, export_notebook=False):
        """ Transpile the tasks from Lemonade's workflow into code """

        using_stdout = out is None
        if using_stdout:
            out = sys.stdout

        ports = {}
        sequential_ports = {}
        counter = 0
        for source_id in graph.edge:
            for target_id in graph.edge[source_id]:
                # Nodes accept multiple edges from same source
                for flow in graph.edge[source_id][target_id].values():
                    flow_id = '[{}:{}]'.format(source_id, flow['source_port'], )

                    if flow_id not in sequential_ports:
                        sequential_ports[flow_id] = \
                            TranspilerUtils.gen_port_name(flow, counter)
                        counter += 1
                    if source_id not in ports:
                        ports[source_id] = {'outputs': [], 'inputs': [],
                                            'named_inputs': {},
                                            'named_outputs': {}}
                    if target_id not in ports:
                        ports[target_id] = {'outputs': [], 'inputs': [],
                                            'named_inputs': {},
                                            'named_outputs': {}}

                    sequence = sequential_ports[flow_id]

                    source_port = ports[source_id]
                    if sequence not in source_port['outputs']:
                        source_port['named_outputs'][
                            flow['source_port_name']] = sequence
                        source_port['outputs'].append(sequence)

                    target_port = ports[target_id]
                    if sequence not in target_port['inputs']:
                        flow_name = flow['target_port_name']
                        # Test if multiple inputs connects to a port
                        # because it may have multiplicity MANY
                        if flow_name in target_port['named_inputs']:
                            if not isinstance(
                                    target_port['named_inputs'][flow_name],
                                    list):
                                target_port['named_inputs'][flow_name] = [
                                    target_port['named_inputs'][flow_name],
                                    sequence]
                            else:
                                target_port['named_inputs'][flow_name].append(
                                    sequence)
                        else:
                            target_port['named_inputs'][flow_name] = sequence
                        target_port['inputs'].append(sequence)

        self.generate_code(graph, job_id, out, params,
                           ports, nx.topological_sort(graph), state,
                           hashlib.sha1(),
                           using_stdout, workflow, deploy, export_notebook)

    def get_data_sources(self, workflow):
        return len(
            [t['slug'] in self.DATA_SOURCE_OPS for t in workflow['tasks']]) == 1


class TranspilerUtils(object):
    """ Utilities for using in Jinja2 related to transpiling and other useful
     functions.
     """

    @staticmethod
    def _get_enabled_tasks_to_execute(instances):
        dependency_controller = DependencyController([])
        result = []
        for instance in TranspilerUtils._get_enabled_tasks(instances):
            task = instance.parameters['task']
            is_satisfied = dependency_controller.is_satisfied(task['id'])
            if instance.must_be_executed(is_satisfied):
                result.append(instance)
        return result

    @staticmethod
    def _get_enabled_tasks(instances):
        return [instance for instance in instances if
                instance.has_code and instance.enabled]

    @staticmethod
    def _get_parent_tasks(instances_map, instance, only_enabled=True):
        if only_enabled:
            dependency_controller = DependencyController([])
            result = []
            for parent_id in instance.parameters['task']['parents']:
                parent = instances_map[parent_id]
                is_satisfied = dependency_controller.is_satisfied(parent_id)
                if is_satisfied and parent.has_code and parent.enabled:
                    method = '{}_{}'.format(
                        parent.parameters['task']['operation']['slug'].replace(
                            '-', '_'), parent.order)
                    result.append((parent_id, method))
            return result
        else:
            return [instances_map[parent_id] for parent_id in
                    instance.parameters['task']['parents']]

    @staticmethod
    def get_ids_and_methods(instances):
        result = OrderedDict()
        for instance in TranspilerUtils._get_enabled_tasks_to_execute(
                instances):
            task = instance.parameters['task']
            result[task['id']] = '{}_{}'.format(
                task['operation']['slug'].replace('-', '_'), instance.order)
        return result

    @staticmethod
    def get_disabled_tasks(instances):
        return [instance for instance in instances if
                not instance.has_code or not instance.enabled]

    @staticmethod
    def get_new_task_id():
        return uuid.uuid1()

    @staticmethod
    def escape_chars(text):
        if isinstance(text, str):
            return text.encode('string-escape').replace('"', '\\"').replace(
                "'", "\\'")
        else:
            return text.encode('unicode-escape').replace('"', '\\"').replace(
                "'", "\\'")

    @staticmethod
    def gen_port_name(flow, seq):
        name = flow.get('source_port_name', 'data')
        parts = name.split()
        if len(parts) == 1:
            name = name[:5]
        elif name[:3] == 'out':
            name = name[:3]
        else:
            name = ''.join([p[0] for p in parts])
        return '{}{}'.format(name, seq)