# -*- coding: utf-8 -*-

import autopep8
import datetime
import hashlib
import inspect
import jinja2
import json
import logging
import networkx as nx
import redis
import sys
import uuid
from collections import OrderedDict
import os
from juicer import operation
from juicer.transpiler import Transpiler, TranspilerUtils, DependencyController
from juicer.spark.transpiler import SparkTranspiler
from juicer.scikit_learn.transpiler import ScikitLearnTranspiler

from juicer import auditing
from juicer.util.jinja2_custom import AutoPep8Extension
from juicer.service import stand_service
from juicer.util.template_util import HandleExceptionExtension
from juicer.multi_platform.operations import OperationModeling


# noinspection SpellCheckingInspection
class MultiPlatformTranspiler(Transpiler):
    """
    Convert Lemonade workflow representation (JSON) into code to be run in
    spark or scikit-learn.
    """

    def __init__(self, configuration, slug_to_op_id=None, port_id_to_port=None):
        super(MultiPlatformTranspiler, self).__init__(
            configuration, os.path.abspath(os.path.dirname(__file__)),
            slug_to_op_id, port_id_to_port)
        self.operations_spark = SparkTranspiler(configuration).operations
        self.operations_sklearn = ScikitLearnTranspiler(configuration).operations
        self.requires_hive = False
        self.requires_hive_warehouse = False
        self.hive_metadata = None

    def get_context(self):
        dict_msgs = {
            'task_completed': _('Task completed'),
            'task_running': _('Task running'),
            'lemonade_task_completed': _('Lemonade task %s completed'),
            'lemonade_task_parents': _('Parents completed, submitting %s'),
            'lemonade_task_started': _('Lemonade task %s started'),
            'lemonade_task_afterbefore': _(
                "Submitting parent task {} before {}")}

        return {'dict_msgs': dict_msgs}

    def get_operation_by_platform(self, operation_slug, platform_id):
        if str(platform_id) == '1':
            class_name = self.operations_spark[operation_slug]
        else:
            class_name = self.operations_sklearn[operation_slug]
        return class_name

    def _assign_operations(self):
        self.operations = {} # TODO

    def generate_code(self, graph, job_id, out, params, ports,
                      sorted_tasks_id, state, task_hash, using_stdout,
                      workflow, deploy=False, export_notebook=False,
                      plain=False):

        # if deploy:
        #     # To be able to convert, workflow must obey all these rules:
        #     # - 1 and exactly 1 data source;
        #     # - Data source must be defined in Limonero with its attributes in
        #     # order to define the schema for data input;
        #     # - For ML models, it is required to have a Save Model operation;
        #     total_ds = 0
        #     for task in workflow['tasks']:
        #         if not task.get('enabled', False):
        #             continue
        #         if task['operation']['slug'] in self.DATA_SOURCE_OPS:
        #             total_ds += 1
        #
        #     if total_ds < 1:
        #         raise ValueError(_(
        #             'Workflow must have at least 1 data source to be deployed.')
        #         )
        #     tasks_ids = reversed(sorted_tasks_id)
        # else:
        tasks_ids = sorted_tasks_id

        instances = OrderedDict()
        transpiler_utils = TranspilerUtils(self)

        audit_events = []
        multi_converter = OperationModeling()
        for i, task_id in enumerate(tasks_ids):
            task = graph.nodes[task_id]['attr_dict']
            task['parents'] = graph.nodes[task_id]['parents']
            self.current_task_id = task_id
            slug_operation = task['operation']['slug']
            platform_target = task['forms']['multiplatform']["value"]['target']
            class_name = self.get_operation_by_platform(slug_operation, platform_target)

            parameters = {}
            not_empty_params = [(k, d) for k, d in
                                list(task['forms'].items()) if d['value']]

            task['forms'] = dict(not_empty_params)
            for parameter, definition in list(task['forms'].items()):
                # @FIXME: Fix wrong name of form category
                # (using name instead of category)
                cat = definition.get('category',
                                     'execution').lower()  # FIXME!!!
                cat = 'paramgrid' if cat == 'param grid' else cat
                cat = 'logging' if cat == 'execution logging' else cat

                if all([cat in ["execution", 'paramgrid', 'param grid',
                                'execution logging', 'logging', 'save',
                                'transformation'],
                        definition['value'] is not None]):

                    task_hash.update(str(definition['value']).encode(
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
                        definition['value'], bool) and not isinstance(
                    definition['value'], int):
                    if '"' in definition['value'] or "'" in definition['value']:
                        if definition['value'][0] != '{':
                            definition['value'] = TranspilerUtils.escape_chars(
                                definition['value'])

            if state is None or state.get(task_id) is None:
                parameters['execution_date'] = None
            else:
                gen_source_code = state.get(task_id, [{}])[0]
                if gen_source_code:
                    parameters['execution_date'] = gen_source_code.get(
                        'execution_date')
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
                'export_notebook': export_notebook,
                'hash': task_hash.hexdigest(),
                'job_id': job_id,
                'operation_id': task['operation']['id'],
                'operation_slug': task['operation']['slug'],
                # Some temporary variables need to be identified by a sequential
                # number, so it will be stored in this field
                'order': i,
                'task': task,
                'task_id': task['id'],
                'transpiler': self,  # Allows operation to notify transpiler
                'transpiler_utils': transpiler_utils,
                'plain': plain,
                'user': workflow['user'],
                'workflow': workflow,
                'workflow_id': workflow['id'],
                # Some operations require the complete workflow data
                'workflow_name': TranspilerUtils.escape_chars(workflow['name']),
            })
            port = ports.get(task['id'], {})
            parameters['parents'] = port.get('parents', [])
            parameters['parents_slug'] = port.get('parents_slug', [])
            parameters['parents_by_port'] = port.get('parents_by_port', [])
            parameters['my_ports'] = port.get('my_ports', [])

            parameters = multi_converter.operations[slug_operation](parameters).convert(platform_target)
            instance = class_name(parameters, port.get('named_inputs', {}),
                                  port.get('named_outputs', {}))

            graph.nodes[task['id']]['is_data_source'] = instance.is_data_source
            parameters['audit_events'] = instance.get_audit_events()

            if self.configuration['juicer'].get('auditing', False):
                audit_events.extend(self.get_audit_info(graph, workflow, task,
                                                        parameters))

            instance.out_degree = graph.out_degree(task_id)
            instances[task['id']] = instance

        if audit_events:

            redis_url = self.configuration['juicer']['servers']['redis_url']
            parsed = urlparse(redis_url)
            redis_conn = redis.Redis(host=parsed.hostname,
                                     port=parsed.port)
            q = Queue(AUDITING_QUEUE_NAME, connection=redis_conn)
            for event in audit_events:
                event['date'] = event['date'].isoformat()
            q.enqueue(AUDITING_JOB_NAME, json.dumps(audit_events))

        # adding information about the parents's multiplicity
        for task_id in instances:
            instances[task_id].parameters['multiplicity'] = dict()
            for p_id in instances[task_id].parameters['task']['parents']:
                for flow in workflow['flows']:
                    target_id = flow['target_id'] 
                    port_name = flow['target_port_name']
                    source_port = flow['source_port']
                    if target_id == task_id and flow['source_id'] == p_id:
                        instances[task_id].parameters['multiplicity'][port_name] = \
                            sum([1 for f in workflow['flows']
                                 if (f['source_port'] == source_port) and (f['source_id'] == p_id)])

        env_setup = {
            'autopep8': autopep8,
            'dependency_controller': DependencyController(
                params.get('requires_info', False)),
            'disabled_tasks': workflow['disabled_tasks'],
            'execute_main': params.get('execute_main', False),
            'instances': list(instances.values()),
            'instances_by_task_id': instances,
            'job_id': job_id,
            'now': datetime.datetime.now(), 'user': workflow['user'],
            'plain': plain,
            'export_notebook': export_notebook,
            'transpiler': transpiler_utils,
            'workflow_name': workflow['name'],
            'workflow': workflow,
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
            out.write(template.render(env_setup))
        elif export_notebook:
            template = template_env.get_template(self.get_notebook_template())
            out.write(template.render(env_setup))
        elif plain:
            template = template_env.get_template(self.get_plain_template())
            out.write(template.render(env_setup))
        else:
            template = template_env.get_template(self.get_code_template())
            gen_source_code = template.render(env_setup)
            if using_stdout:
                out.write(gen_source_code)
            else:
                out.write(gen_source_code)
            stand_config = self.configuration.get('juicer', {}).get(
                'services', {}).get('stand')
            if stand_config and job_id:
                # noinspection PyBroadException
                try:
                    stand_service.save_job_source_code(
                        stand_config['url'], stand_config['auth_token'], job_id,
                        gen_source_code)
                except Exception as ex:
                    log.exception(str(ex))
