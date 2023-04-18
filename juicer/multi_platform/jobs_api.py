# coding=utf-8

import pymysql.cursors
import pymysql
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
from collections import OrderedDict
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import datetime
from gettext import gettext

#LIBRARY_PATH = os.path.expanduser("/home/lucasmsp/workspace/bigsea/docker-lemonade/juicer/")
#sys.path.insert(0, LIBRARY_PATH)
#os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + LIBRARY_PATH

from juicer.compss.transpiler import COMPSsTranspiler
from juicer.keras.transpiler import KerasTranspiler
from juicer.scikit_learn.transpiler import ScikitLearnTranspiler
from juicer.runner import configuration
from juicer.service.tahiti_service import query_tahiti
from juicer.spark.transpiler import SparkTranspiler
from juicer.workflow.workflow import Workflow
from juicer.service import tahiti_service, limonero_service
from juicer.transpiler import TranspilerUtils
import traceback

from juicer.multi_platform.resources_api import Cluster
from juicer.multi_platform.limonero_api import Dataset
from juicer.multi_platform.operations import OperationModeling
from juicer.multi_platform.transpiler import MultiPlatformTranspiler
# !pip3 install pymysql

from juicer.multi_platform.auxiliar_services import LIMONERO_DB, STAND_DB
DEBUG = False


def log(msg):
    if DEBUG:
        print(msg)


class LemonadeJob(object):
    action_operator = []

    ERROR_UNKNOWN_SIZE = "[WARN] Skipping lemonade job because one of it's input data size is unknown ({} - id {})."
    ERROR_CONTAINS_OLD_OPERATION = "[WARN] Skipping lemonade job because it contains an operation " \
                                   "that does not exist anymore. "
    ERROR_PLATFORM_NOT_SUPPORTED = "[ERROR] Platform not supported!"
    ERROR_SLUG_NOT_FOUND = "[ERROR] Slug unknown."

    # vc so teria controle se chamasse o has_code() de cada metodo
    ERROR_TASK_NOT_EXECUTED = "[WARN] Skipping lemonade job because it not contains any executed task."

    def __init__(self, workflow, actions, juicer_config, slugs_operation, mode="break-by-actions"):
        self.slugs_operation = slugs_operation
        self.config = juicer_config
        self.action_operator = actions
        self.graph = nx.MultiDiGraph()
        self.workflow = workflow
        self.tasks = self.workflow['tasks']
        self.stand = {}
        self.operation_api = OperationModeling()

        self.platform = None
        self.cluster = None
        self.job_id = None
        self.df_states = {}

        # Tasks disabled do not execute
        self.disabled_tasks = {}

        self._build_initial_workflow_graph()
        self._update_workflow_graph_adding_parameters()

        if mode == "break-by-actions":
            self._update_workflow_graph_adding_actions()
        elif mode == "break-by-stages":
            self._update_workflow_graph_adding_stages()
        elif mode == "break-by-flow":
            self._update_workflow_graph_adding_only_final_actions()
        elif mode == "break-by-tasks":
            self._update_workflow_graph_adding_action_to_each_task()
        elif mode == "complete":
            self._update_workflow_graph_adding_only_one_action()
        else:
            raise Exception("Mode not supported.")

        self.jobs = {}
        self._split_dag_jobs()

        self.created_timestamp = None
        self.scheduled_timestamp = None
        self.finished_timestamp = None
        self.started_timestamp = None
        self.seconds_to_start = None

    def plot(self, subset=None):
        labels = {}
        color_map = []

        if subset:
            g = self.graph.subgraph(subset)

        else:
            g = self.graph

        for task_id in g.nodes():
            label = g.nodes[task_id].get("operation", {}).get("slug", None)
            if not label:
                label = g.nodes[task_id].get("label", "")
            labels[task_id] = label
            color = "blue"
            if "action" in task_id:
                color = "red"
            elif self._is_task_input_source(task_id):
                color = "green"
            color_map.append(color)

        pos = graphviz_layout(g, prog='dot')
        nx.draw(g, node_color=color_map, labels=labels, with_labels=True, pos=pos)
        plt.show()

    def get_jobs(self):
        return self.jobs

    @staticmethod
    def is_enabled(t):
        return t.get('enabled', True) and t.get('environment', 'DESIGN') == 'DESIGN'

    def is_action(self, task, operation_id):
        """
        Será uma ação se uma operação estiver na lista de ações, se gerar uma amostra, 
        ou se tiver "dc_w" no nome e pedir pra gerar um schema. 
        """
        cond1 = int(task['forms'].get("display_sample", {"value": "0"})["value"]) == 1 \
                or (("dc_w" in self.workflow["name"]) and
                    int(task['forms'].get("display_schema", {"value": "0"})["value"]) == 1)
        cond2 = operation_id in self.action_operator
        return any([cond1, cond2])

    def _get_operations(self, workflow_id):
        """ Returns operations available in Tahiti """
        tahiti_conf = self.config['juicer']['services']['tahiti']
        params = {
            'base_url': tahiti_conf['url'],
            'item_path': 'operations',
            'token': str(tahiti_conf['auth_token']),
            'item_id': '',
            'qs': 'workflow={}'.format(workflow_id)
        }

        # Querying tahiti operations to get number of inputs and outputs
        return tahiti_service.query_tahiti(
            params['base_url'], params['item_path'], params['token'],
            params['item_id']).get('data')

    def _build_initial_workflow_graph(self):
        """ Builds a graph with the tasks """

        operations_tahiti = {op['id']: op
                             for op in self._get_operations(self.workflow['id'])}
        # Querying all operations from tahiti one time
        task_map = {}

        for task in self.workflow['tasks']:

            if task.get('enabled', True) and (task.get('environment', 'DESIGN') == 'DESIGN'):
                operation = operations_tahiti.get(task['operation']['id'])

                # se a operacao ainda existe, senao, desconsidere esse fluxo
                if not operation:
                    raise Exception(self.ERROR_CONTAINS_OLD_OPERATION)

                form_fields = {}
                for form in operation.get('forms', []):
                    for field in form['fields']:
                        form_fields[field['name']] = form['category']

                task_map[task['id']] = {'task': task, 'operation': operation}
                if operation and (operation['slug'] not in ["comment",
                                                            "publish-as-visualization",
                                                            "load-model",
                                                            "save-model"]):
                    # Slug information is required in order to select which
                    # operation will be executed
                    task['operation']['slug'] = operation['slug']
                    task['operation']['name'] = operation['name']
                    task['operation']['ports'] = dict(
                        [
                            (
                                port['id'],
                                {
                                    'id': port['id'],
                                    'type': port['type'],
                                    'slug': port['slug'],
                                    'multiplicity': port['multiplicity'],
                                    'interfaces': [pi['name'] for pi in port[
                                        'interfaces']]
                                }) for port in operation['ports']])
                    ports_list = operation['ports']
                    # Get operation requirements in tahiti
                    result = {
                        'N_INPUT': 0,
                        'N_OUTPUT': 0,
                        'PORT_NAMES': [],
                        'M_INPUT': 'None',
                        'M_OUTPUT': 'None'
                    }

                    # Correct form field types if the interface (Citron) does
                    # not send this information
                    for k, v in list(task.get('forms', {}).items()):
                        v['category'] = form_fields.get(k, 'EXECUTION')

                    for port in ports_list:
                        if port['type'] == 'INPUT':
                            result['M_INPUT'] = port['multiplicity']
                            if 'N_INPUT' in result:
                                result['N_INPUT'] += 1
                            else:
                                result['N_INPUT'] = 1
                        elif port['type'] == 'OUTPUT':
                            result['M_OUTPUT'] = port['multiplicity']
                            if 'N_OUTPUT' in result:
                                result['N_OUTPUT'] += 1
                            else:
                                result['N_OUTPUT'] = 1
                            if 'PORT_NAMES' in result:
                                result['PORT_NAMES'].append(
                                    (int(port['order']), port['name']))
                            else:
                                result['PORT_NAMES'] = [
                                    (int(port['order']), port['name'])]

                    port_names = [kv[1] for kv in sorted(
                        result['PORT_NAMES'], key=lambda _kv: _kv[0])]
                    task['port_names'] = port_names
                    self.graph.add_node(
                        task.get('id'),
                        name=task.get('name'),
                        forms=task.get('forms'),
                        operation=task.get('operation'),
                        in_degree_required=result['N_INPUT'],
                        in_degree_multiplicity_required=result['M_INPUT'],
                        out_degree_required=result['N_OUTPUT'],
                        out_degree_multiplicity_required=result['M_OUTPUT'],
                        port_names=port_names,
                        parents=[],
                        attr_dict=task)
                else:
                    self.disabled_tasks[task['id']] = task

            else:
                self.disabled_tasks[task['id']] = task

        for flow in self.workflow['flows']:

            # Ignore disabled tasks
            if all([flow['source_id'] not in self.disabled_tasks,
                    flow['target_id'] not in self.disabled_tasks]):
                # Updates the source_port_name and target_port_name. They are
                # used in the transpiler part instead of the id of the port.
                source_port = list([p for p in
                                    task_map[flow['source_id']]['operation'][
                                        'ports'] if
                                    int(p['id']) == int(flow['source_port'])])

                target_port = list([p for p in
                                    task_map[flow['target_id']]['operation'][
                                        'ports'] if
                                    int(p['id']) == int(flow['target_port'])])

                if all([source_port, target_port]):
                    flow['target_port_name'] = target_port[0]['slug']
                    flow['source_port_name'] = source_port[0]['slug']

                    self.graph.add_edge(flow['source_id'], flow['target_id'],
                                        attr_dict=flow)
                    self.graph.nodes[flow['target_id']]['parents'].append(
                        flow['source_id'])
                else:
                    print(
                        _("Incorrect configuration for ports: %s, %s"),
                        source_port, target_port)
                    raise ValueError(_(
                        "Invalid or non-existing port: '{op}' {s} {t}").format(
                        op=task_map[flow['source_id']]['operation']['name'],
                        s=flow['source_port'], t=flow['target_port']))

    def _update_workflow_graph_adding_actions(self):
        tasks_ids = reversed(list(nx.topological_sort(self.graph)))
        actions = []
        for i, task_id in enumerate(tasks_ids):
            task = self.graph.nodes[task_id]['attr_dict']
            operation_id = task['operation']['id']

            if self.is_enabled(task) and self.is_action(task, operation_id):
                t = "action-{}".format(len(actions))
                # operation_slug = task['operation']["slug"]
                # print(f"Task {task_id} ({operation_slug}) is enabled and is an action {t}")

                actions.append(t)
                self.graph.add_node(t,
                                    task_id=task.get('id'),
                                    operation_id=operation_id,
                                    label=t,
                                    forms=None,
                                    attr_dict={'parent': task.get('id'), "forms": None})

                out_edges = [out_node for in_node, out_node in list(self.graph.out_edges([task_id]))]
                self.graph.add_edge(task_id, t)

                if len(out_edges) > 0:
                    for out_node in out_edges:
                        if "action" not in out_node:
                            self.graph.remove_edge(task_id, out_node)
                            self.graph.add_edge(t, out_node)

    def _update_workflow_graph_adding_action_to_each_task(self):
        tasks_ids = reversed(list(nx.topological_sort(self.graph)))
        actions = []
        for i, task_id in enumerate(tasks_ids):
            task = self.graph.nodes[task_id]['attr_dict']
            operation_id = task['operation']['id']

            if self.is_enabled(task):
                t = "action-{}".format(len(actions))
                # operation_slug = task['operation']["slug"]
                # print(f"Task {task_id} ({operation_slug}) is enabled and is an action {t}")

                actions.append(t)
                self.graph.add_node(t,
                                    task_id=task.get('id'),
                                    operation_id=operation_id,
                                    label=t,
                                    forms=None,
                                    attr_dict={'parent': task.get('id'), "forms": None})

                out_edges = [out_node for in_node, out_node in list(self.graph.out_edges([task_id]))]
                self.graph.add_edge(task_id, t)

                if len(out_edges) > 0:
                    for out_node in out_edges:
                        if "action" not in out_node:
                            self.graph.remove_edge(task_id, out_node)
                            self.graph.add_edge(t, out_node)

    def _update_workflow_graph_adding_stages(self):
        tasks_ids = reversed(list(nx.topological_sort(self.graph)))
        actions = []
        for i, task_id in enumerate(tasks_ids):
            task = self.graph.nodes[task_id]['attr_dict']
            operation_id = task['operation']['id']

            cond1 = int(task['forms'].get("display_sample", {"value": "0"})["value"]) == 1

            cond2 = self.graph.nodes[task_id]["stage_behavior"] == "split-stage"
            if self.is_enabled(task) and (cond1 or cond2):
                t = "action-{}".format(len(actions))

                actions.append(t)
                self.graph.add_node(t,
                                    task_id=task.get('id'),
                                    operation_id=operation_id,
                                    label=t,
                                    forms=None,
                                    attr_dict={'parent': task.get('id'), "forms": None})

                out_edges = [out_node for in_node, out_node in list(self.graph.out_edges([task_id]))]
                self.graph.add_edge(task_id, t)

                if len(out_edges) > 0:
                    for out_node in out_edges:
                        if "action" not in out_node:
                            self.graph.remove_edge(task_id, out_node)
                            self.graph.add_edge(t, out_node)

    def _update_workflow_graph_adding_only_one_action(self):
        tasks_ids = reversed(list(nx.topological_sort(self.graph)))
        actions = []

        t = "action-{}".format(len(actions))
        actions.append(t)

        self.graph.add_node(t,
                            task_id=-1,
                            operation_id=-1,
                            label=t,
                            forms=None,
                            attr_dict={'parent': -1, "forms": None})

        for i, task_id in enumerate(tasks_ids):
            task = self.graph.nodes[task_id]['attr_dict']
            operation_id = task['operation']['id']

            if self.is_enabled(task) and self.is_action(task, operation_id):

                out_edges = [out_node for in_node, out_node in list(self.graph.out_edges([task_id]))]
                if len(out_edges) == 0:
                    self.graph.add_edge(task_id, t)

    def _update_workflow_graph_adding_only_final_actions(self):
        tasks_ids = reversed(list(nx.topological_sort(self.graph)))
        actions = []
        for i, task_id in enumerate(tasks_ids):
            task = self.graph.nodes[task_id]['attr_dict']
            operation_id = task['operation']['id']

            if self.is_enabled(task) and self.is_action(task, operation_id):
                t = "action-{}".format(len(actions))
                # operation_slug = task['operation']["slug"]
                # print(f"Task {task_id} ({operation_slug}) is enabled and is an action {t}")

                out_edges = [out_node for in_node, out_node in list(self.graph.out_edges([task_id]))]
                if len(out_edges) == 0:
                    actions.append(t)
                    self.graph.add_node(t,
                                        task_id=task.get('id'),
                                        operation_id=operation_id,
                                        label=t,
                                        forms=None,
                                        attr_dict={'parent': task.get('id'), "forms": None})

                    self.graph.add_edge(task_id, t)

    def _update_workflow_graph_adding_parameters(self):
        tasks_ids = nx.topological_sort(self.graph)

        for i, task_id in enumerate(tasks_ids):
            task = self.graph.nodes[task_id]['attr_dict']
            task['parents'] = self.graph.nodes[task_id]['parents']
            slug_operation = task['operation']['slug']

            parameters = {}
            not_empty_params = [(k, d) for k, d in
                                list(task['forms'].items()) if d['value']]
            task['forms'] = dict(not_empty_params)

            for parameter, definition in list(task['forms'].items()):
                cat = definition.get('category', 'execution').lower()
                cat = 'paramgrid' if cat == 'param grid' else cat
                cat = 'logging' if cat == 'execution logging' else cat

                if all([cat in ["execution", 'paramgrid', 'param grid',
                                'execution logging', 'logging', 'save',
                                'transformation'],
                        definition['value'] is not None]):

                    if cat in ['paramgrid', 'logging']:
                        if cat not in parameters:
                            parameters[cat] = {}
                        parameters[cat][parameter] = definition['value']
                    else:
                        parameters[parameter] = definition['value']

                # escape invalid characters for code generation
                # except JSON (starting with {)
                if definition['value'] is not None and \
                        not isinstance(definition['value'], bool) and \
                        not isinstance(definition['value'], int):
                    if '"' in definition['value'] or "'" in definition['value']:
                        if definition['value'][0] != '{':
                            definition['value'] = TranspilerUtils.escape_chars(definition['value'])

            true_values = (1, '1', True, 'true', 'True')

            parameters.update({
                'display_sample': task['forms'].get('display_sample', {}).get('value') in true_values,
                'display_schema': task['forms'].get('display_schema', {}).get('value') in true_values,
                'operation_id': task['operation']['id'],
                'operation_slug': slug_operation,
                'order': i,
                'task': task,
                'task_id': task['id'],
            })
            self.graph.nodes[task_id]["parameters"] = parameters
            try:
                operation = self.operation_api.all_operations[slug_operation](parameters)
                self.graph.nodes[task_id]["object"] = operation
                self.graph.nodes[task_id]["stage_behavior"] = operation.stage_behavior()
            except Exception:
                raise Exception(self.ERROR_CONTAINS_OLD_OPERATION)

    def print_path(self, tasks):
        tmp = ""
        for t in tasks:
            if "job" in t:
                tmp += "{}, ".format(t[0:8])
            else:
                tmp += "{} ({}), ".format(self.graph.nodes[t].get('operation', {"slug": "action"})['slug'], t[0:8])
        print("Path: ", tmp[:-2])

    def condition_to_merge_paths(self, g1, g2, g3):
        """
        A interseção tem que: ter pelo menos uma ação; nao pode ser subconjunto do outro; e ser maior que um
        (pois a acao nao conta); os itens que se interseptam nao pode ter out-degree > 1 (senao e' uma bifurcacao)
        :param g1:
        :param g2:
        :param g3:
        :return:
        """
        nodes3 = g3.nodes()
        nodes1 = g1.nodes()
        nodes2 = g2.nodes()

        if len(nodes3) <= 1:
            log("condition_to_merge_paths: cond 1")
            return False
        elif not any([True for node in nodes3 if "action" in node]):
            log("condition_to_merge_paths: cond 2")
            return False
        elif set(nodes3) == set(nodes2) or set(nodes3) == set(nodes1):
            log("condition_to_merge_paths: cond 3")
            return False
        elif any([True for n in nodes3 if len(list(self.graph.out_edges([n]))) > 1]):
            return False
        else:
            return True

    def convert_subset2job(self, ppath, i, job_id):
        #log("Checking if new job is a subset from others paths")
        path = ppath[i]
        for j in range(len(ppath)):
            if j != i:
                j_path = ppath[j]
                if len(list(set(j_path).intersection(set(path)))) == len(path):
                    ppath[j] = list(set(j_path)-set(path)) + [job_id]
                    #log("New Path: "+str(ppath[j]))
        return ppath

    def _is_task_input_source(self, task_id):
        input_sources = ["execute-python", "data-reader"]  # TODO
        slug = self.graph.nodes[task_id].get("operation", {"slug": "action"})["slug"]
        return slug in input_sources and list(self.graph.in_degree([task_id]))[0][1] == 0

    def _split_dag_jobs(self):
        """
        Split the DAG into several jobs.
        :return:
        """

        startpoints = []
        for node in self.graph.nodes():
            if self._is_task_input_source(node):
                startpoints.append(node)
        endpoints = [node for node in self.graph.nodes() if "action" in node]

        # Define pontos de partidas (caixas sem input-data) e actions
        # para encontrar os caminhos.
        ppath = []
        for sp in startpoints:
            for ep in endpoints:
                # log("Finding path from {} to {}".format(sp, ep))
                try:
                    path = sorted(list(nx.all_shortest_paths(self.graph, sp, ep)), key=len)
                    # log("Path from {} to {} found".format(sp, ep))
                    # se dois caminhos tem o mesmo ponto de origem e de destino, entao fazem parte obrigatoriamente
                    # do mesmo job
                    path = list(set([x for xs in path for x in xs]))
                    ppath.append(path)
                except:
                    log("Path from {} to {} does not exist".format(sp, ep))

        log("PPath (caminhos) = " + str(len(ppath)))

        # Avalia se os caminhos se interseptam (por exemplo, a partir de um join) para juntar-los em jobs
        for (i, j) in list(itertools.combinations(range(len(ppath)), 2)):
            log(f"Checking Path {i} and {j} if can be merged.")
            g1 = self.graph.subgraph(ppath[i])
            g2 = self.graph.subgraph(ppath[j])
            g3 = nx.intersection(g1, g2)

            if self.condition_to_merge_paths(g1, g2, g3):
                ppath[i] = list(set(ppath[i])) + list(set(ppath[j]))
                ppath[j] = ppath[i]
                log(f"Path {i} and {j} was merged.")

        # remove merged paths
        ppath.sort()
        ppath = list(k for k, _ in itertools.groupby(ppath))

        # sorting by length
        ppath.sort(key=len)

        # Descobre se um caminho é subconjunto de outro. Se for, esse pode ser considerado um job nessa fase
        for (i, j) in list(itertools.combinations(range(len(ppath)), 2)):

            g1 = self.graph.subgraph(ppath[i])
            g2 = self.graph.subgraph(ppath[j])
            g3 = nx.intersection(g1, g2)

            # Se A for subconjunto de B, transforme A em um job
            n1 = len(g1.nodes())
            n2 = len(g2.nodes())
            n3 = len(g3.nodes())

            if n3 > 0:
                if n1 == n3:
                    job_id = "job-{:02d}".format(len(self.jobs))
                    self.jobs[job_id] = {"id": i}
                    ppath = self.convert_subset2job(ppath, i, job_id)
                elif n2 == n3:
                    job_id = "job-{:02d}".format(len(self.jobs))
                    self.jobs[job_id] = {"id": j}
                    ppath = self.convert_subset2job(ppath, j, job_id)

        # Na lista de jobs, adiciona os Paths que não foram juntados na etapa anterior
        for i, p in enumerate(ppath):
            found = False
            for j in list(self.jobs.keys()):
                if i == self.jobs[j]["id"]:
                    found = True
            if not found:
                job_id = "job-{:02d}".format(len(self.jobs))
                self.jobs[job_id] = {"id": i}

        for j in list(self.jobs.keys()):
            id_path = self.jobs[j]["id"]
            tasks = ppath[id_path]
            self.jobs[j]["tasks"] = tasks
            if len(self.jobs[j]["tasks"]) == 1:
                self.jobs.pop(j, None)

    def get_dataflow(self, v="v5"):

        if v == "v5":
            return self._gen_dataflow_v5()

    def find_jobs_order(self):
        tasks = re.findall("task_futures\[\'.*\] = ", self.stand.get('source_code', ""))
        orders = []
        if tasks:
            for i, t in enumerate(tasks):
                t = t[14:-5]
                for j, job in enumerate(self.jobs):
                    tasks = self.jobs[job]["tasks"]
                    if t in str(tasks):
                        if job not in orders:
                            orders.append(job)
        else:
            orders = sorted(list(self.jobs.keys()))
        return orders

    def calculate_outcome_v4(self, input_size, operations_dict):
        base_value = 1.0
        factors = operations_dict.get("data_amplification", [])
        for f in factors:
            base_value += f
        return input_size * base_value

    def _gen_dataflow_v5(self):

        task_orders = self.find_jobs_order()

        scenario_idx = 8        
        operations_idx = 9
        job_parent_idx = 2
        time_idx = 4
        task_idx = 3

        dataflow = [0 for _ in range(len(self.jobs))]
        if self.cluster:
            n_executors = self.cluster.executors
            n_cores = n_executors * self.cluster.executor_cores
            ram = n_executors * self.cluster.executor_memory
        else:
            n_cores = -1
            ram = -1

        for order_job, job_id in enumerate(task_orders):
            try:
                self.jobs[job_id]["cluster"] = self.cluster
                self.jobs[job_id]["order"] = order_job
            except e:
                print(job_id)
                print(task_orders)

            dataflow[order_job] = [self.job_id,                             # job id
                                   order_job+1,                             # job_order
                                   "",                                      # job_parent
                                   "",                                      # task_id
                                   -1,                                      # time in seconds
                                   self.platform,                           # platform_id
                                   n_cores,                                 # number of avaliable cores
                                   ram,                                     # available RAM
                                   "",                                      # scenario
                                   {}                                       # operations
                                   ]

            tasks = self.jobs[job_id]["tasks"]
            sorted_tasks = list(nx.topological_sort(self.graph.subgraph(tasks)))

            for t in tasks:
                if "action" in t:
                    self.jobs[job_id]["action"] = self.graph.nodes[t]['task_id']

            for i, task_id in enumerate(sorted_tasks):
                task = self.graph.nodes[task_id]
                slug = task.get("operation", {"slug": None})["slug"]
                
                dataflow[order_job][task_idx] = sorted_tasks[i-1]

                if slug:
                    db = None
                    try:
                        operation = task["object"]
                    except Exception as e:
                        raise Exception(self.ERROR_SLUG_NOT_FOUND)

                    if slug == "data-reader":
                        base_id = int(task["parameters"]['data_source'])
                        db = Dataset(base_id)
                        db_size = db.disk_size
                        if db_size < 0:
                            raise Exception(self.ERROR_UNKNOWN_SIZE.format(db.name, db.base_id))
                        # dataflow[order_job][size_idx][base_id] = db
                        new_dfs = operation.estimate_output([db.stats])
                        self.df_states[task_id] = new_dfs
                        self.graph.nodes[task_id]["model"] = new_dfs
                    else:
                        dfs = []
                        try:
                            for parent_task_id in task["parents"]:
                                dfs.append(self.df_states[parent_task_id][0])
                        except Exception as e:
                            print("ERROR")
                            print("task_id", task_id)
                            print("slug:", slug)
                            print(self.df_states)
                            print(sorted_tasks)
                            print(self.jobs)
                            raise e
                        
                        # TODO: suporte ao split
                        # TODO: manter ordem {"input1 ..."}
                        new_dfs = operation.estimate_output(dfs)
                        self.df_states[task_id] = new_dfs
                        self.graph.nodes[task_id]["model"] = new_dfs

                    
                    dataflow[order_job][operations_idx][slug] = operation
                    dataflow[order_job][scenario_idx] = slug

        return dataflow


class HistoricalLemonadeJob(LemonadeJob):

    def __init__(self, stand, job_log, actions, juicer_config, slugs_operation, mode):

        self.stand = stand
        workflow = json.loads(stand["workflow_definition"])
        super().__init__(workflow, actions, juicer_config, slugs_operation, mode)

        self.cluster = Cluster(int(stand['cluster_id']))
        self.platform = int(self.workflow['platform']['id'])
        self.job_id = stand['id']
        self.job_log = job_log

        self.created_timestamp = stand["created"]
        self.scheduled_timestamp = stand["started"]
        self.finished_timestamp = stand["finished"]
        self.started_timestamp = min([j["l_date"] for j in self.job_log])
        self.seconds_to_start = (self.started_timestamp - self.scheduled_timestamp).total_seconds()
        self.real_time = (self.finished_timestamp - self.started_timestamp).total_seconds()

        self.original_transpiler = self.get_transpiler(self.platform)  # TODO
        self._inflate_jobs_info()

    def summarize_job(self):
        print("""
        Job: {} | Workflow: {} | Platform: {} | Cluster: {}
        Created Timestamp: {}
        Scheduled Timestamp: {}
        Started Timestamp: {}
        Finished Timestamp: {}
        """.format(self.job_id, self.workflow["id"], self.platform, self.cluster.cluster_id,
                   self.created_timestamp, self.scheduled_timestamp, self.started_timestamp,
                   self.finished_timestamp))

    def get_transpiler(self, platform_id):
        if platform_id == 1:
            transpiler = SparkTranspiler(None)
            log("Set original transpiler as Spark")
        elif platform_id == 4:
            transpiler = ScikitLearnTranspiler(None)
            log("Set original transpiler as Sklearn")
        elif platform_id == 6:
            # TODO
            transpiler = MultiPlatformTranspiler(None)
            log("Set original transpiler as MultiPlatform")
        else:
            raise Exception(self.ERROR_PLATFORM_NOT_SUPPORTED)
        return transpiler

    def _inflate_jobs_info(self):
        """
        Para cada job, crie a definição das variaveis mínimas: recursos, operações e tempo gasto.
        :return:
        """
        jobs_inflated = []

        # adicao do tempo gasto
        while len(jobs_inflated) < len(self.jobs):
            for job_id in list(self.jobs.keys()):
                if job_id not in jobs_inflated:
                    tasks = self.jobs[job_id]["tasks"]
                    dependencies = [j for j in tasks if "job-" in j]
                    if len(dependencies) == 0:
                        self.get_timestamps(job_id)
                        jobs_inflated.append(job_id)
                    elif not all([False for dep in dependencies if dep in jobs_inflated]):
                        offset = sum([self.jobs.get(dep, {}).get("total_seconds", 0)
                                      for dep in dependencies])
                        self.get_timestamps(job_id, offset)
                        jobs_inflated.append(job_id)

    def _get_duration_task(self, task_id):
        s1 = None
        s2 = None

        for task2 in self.job_log:
            if task_id == task2["task_id"]:
                if (task2["l_status"] == "RUNNING") and ("converted" not in task2['message']):
                    if s1:
                        s1 = min([s1, task2['l_date']])
                    else:
                        s1 = task2['l_date']

                elif task2["l_status"] == "COMPLETED":
                    if s2:
                        s2 = max([s2, task2['l_date']])
                    else:
                        s2 = task2['l_date']
        try:
            duration = (s2 - s1).total_seconds()
        except Exception as e:
            duration = -1

        return duration

    def _get_timestamp_as_pandas(self, job_id):
        """
        Considera exatamente o tempo de execuçao do trecho pelo Lemonade.
        O calculo e feito a partir da subtraçao do maior timestamp pelo menor

        :param job_id:
        :return:
        """
        tasks = self.jobs[job_id]["tasks"]

        duration = 0.0
        for task in tasks:
            if "job" not in task and "action" not in task:
                duration += self._get_duration_task(task)
        return duration

    def _get_timestamp_as_spark(self, job_id, offset=0):

        tasks = self.jobs[job_id]["tasks"]

        duration = 0.0
        for task in tasks:
            if "job" not in task and "action" not in task:
                duration += self._get_duration_task(task)

        if duration >= 0.0:
            duration -= offset
            if duration <= 0.0:
                duration = 0.01

        return duration

    def get_timestamps(self, job_id, offset=0):

        if self.platform == 1:
            duration = self._get_timestamp_as_spark(job_id, offset)
        else:
            duration = self._get_timestamp_as_pandas(job_id)

        if duration == -1:
            # se for -1, significa que esse fluxo nao foi executado, entao devemos remover esse flow
            self.jobs.pop(job_id)
        else:
            if duration == 0.0:
                duration = 0.01
            self.jobs[job_id]["total_seconds"] = duration

    def _gen_dataflow_v5(self):

        task_orders = self.find_jobs_order()
        operations_idx = 9
        job_parent_idx = 2
        time_idx = 4
        task_idx = 3

        if "dc_w" not in self.workflow["name"]:
            return 

        scenario = self.workflow["name"].split("_")[1][1:]

        # TODO: 
        if "cleaning-mode" in scenario:
            scenario = "clean-missing"
        elif "svm" in scenario:
            scenario = "svm-classification-model"
        elif "filter" in scenario:
            scenario = "filter-selection"
        
        dataflow = [0 for _ in range(len(self.jobs))]
        if self.cluster:
            n_executors = self.cluster.executors
            n_cores = n_executors * self.cluster.executor_cores
            ram = n_executors * self.cluster.executor_memory
        else:
            n_cores = -1
            ram = -1

        for order_job, job_id in enumerate(task_orders):
            try:
                self.jobs[job_id]["cluster"] = self.cluster
                self.jobs[job_id]["order"] = order_job
            except e:
                print(job_id)
                print(task_orders)
                

            dataflow[order_job] = [self.job_id,                             # job id
                                   order_job+1,                             # job_order
                                   "",                                      # job_parent
                                   "",                                      # task_id
                                   self.jobs[job_id].get("total_seconds"),  # time in seconds
                                   self.platform,                           # platform_id
                                   n_cores,                                 # number of avaliable cores
                                   ram,                                     # available RAM
                                   scenario,                                # scenario
                                   {}                                       # operations
                                   ]

            tasks = self.jobs[job_id]["tasks"]
            sorted_tasks = list(nx.topological_sort(self.graph.subgraph(tasks)))

            for t in tasks:
                if "action" in t:
                    self.jobs[job_id]["action"] = self.graph.nodes[t]['task_id']

            tasks_to_remove = set()
            for task_id in sorted_tasks:
                task = self.graph.nodes[task_id]
                slug = task.get("operation", {"slug": None})["slug"]

                dataflow[order_job][task_idx] = task_id

                if slug:
                    db = None
                    try:
                        operation = task["object"]
                    except e:
                        raise Exception(self.ERROR_SLUG_NOT_FOUND)

                    # computa apenas as tarefas de interesse. Tudo que estiver em
                    # tasks_to_remove, será removido do tempo de execução do job
                    if scenario in slug:
                        dataflow[order_job][operations_idx][slug] = operation
                    elif (scenario == 'data-migration') and (slug == 'data-writer'):
                        p = operation.parameters
                        new_op = self.operation_api.all_operations[scenario](p)
                        dataflow[order_job][operations_idx][scenario] = new_op
                        tasks_to_remove.add(task_id)
                        dataflow[order_job][time_idx] = self.real_time
                    else:
                        tasks_to_remove.add(task_id)

                    if slug == "data-reader":
                        base_id = int(task["parameters"]['data_source'])
                        db = Dataset(base_id)
                        db_size = db.disk_size
                        if db_size < 0:
                            raise Exception(self.ERROR_UNKNOWN_SIZE.format(db.name, db.base_id))
                        # dataflow[order_job][size_idx][base_id] = db
                        new_dfs = operation.estimate_output([db.stats])
                        self.df_states[task_id] = new_dfs
                        self.graph.nodes[task_id]["model"] = new_dfs
                    else:
                        dfs = []
                        for p in task["parents"]:
                            dfs.append(self.df_states[p][0])

                        # TODO: suporte ao split
                        new_dfs = operation.estimate_output(dfs)
                        self.graph.nodes[task_id]["model"] = new_dfs
                        self.df_states[task_id] = new_dfs
                        if (scenario == 'data-migration') and (slug == 'data-writer'):
                            dataflow[order_job][operations_idx][scenario].estimate_output(dfs)

            # remove data-reader time
            for task in tasks_to_remove:
                dataflow[order_job][time_idx] -= self._get_duration_task(task)


        # for row, job_id in enumerate(task_orders):
        #     for tasks_id in self.jobs[job_id]["tasks"]:
        #         if "job-" in tasks_id:
        #             recent_parent = self.jobs[tasks_id]["order"]
        #             dataflow[row][job_parent_idx] += str(recent_parent + 1)+"_"
        #             # for k, v in dataflow[recent_parent][output_size_idx].items():
        #             #     dataflow[row][size_idx][k] = v

            #dataflow[row][output_size_idx] = self.calculate_outcome_v4(dataflow[row][size_idx],
            #                                                           dataflow[row][operations_idx])
        return dataflow
