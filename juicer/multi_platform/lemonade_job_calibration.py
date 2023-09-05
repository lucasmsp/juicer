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
import copy
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
from juicer.multi_platform.lemonade_job import LemonadeJob
DEBUG = False


def log(msg):
    if DEBUG:
        print(msg)


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
        self.first_timestamp = min([j["MIN"] for j in self.job_log]) # primeiro timestamp de uma operação
        self.last_timestamp = max([j["MAX"] for j in self.job_log]) # primeiro timestamp de uma operação
        
        self.seconds_to_start = (self.first_timestamp - self.scheduled_timestamp).total_seconds()
        self.execution_time = (self.last_timestamp - self.first_timestamp).total_seconds()

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
                   self.created_timestamp, self.scheduled_timestamp, self.first_timestamp,
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
        # s1 = None
        # s2 = None

        duration = 0.01
        #print(self.job_log)
        for task2 in self.job_log:
            if task_id == task2["task_id"]:
                duration = max([task2['seconds'], 0.01])
#                 if (task2["l_status"] == "RUNNING") and ("converted" not in task2['message']):
#                     if s1:
#                         s1 = min([s1, task2['l_date']])
#                     else:
#                         s1 = task2['l_date']

#                 elif task2["l_status"] == "COMPLETED":
#                     if s2:
#                         s2 = max([s2, task2['l_date']])
#                     else:
#                         s2 = task2['l_date']
#         try:
#             duration = (s2 - s1).total_seconds()
#         except Exception as e:
#             duration = -1

        return duration

#     def _get_timestamp_as_pandas(self, job_id):
#         """
#         Considera exatamente o tempo de execuçao do trecho pelo Lemonade.
#         O calculo e feito a partir da subtraçao do maior timestamp pelo menor

#         :param job_id:
#         :return:
#         """
#         tasks = self.jobs[job_id]["tasks"]

#         duration = 0.0
#         for task in tasks:
#             if ("job" not in task) and ("action" not in task):
#                 d = self._get_duration_task(task)
#                 # print(f"{task}: {d}s")
#                 duration += d
#         return duration

#     def _get_timestamp_as_spark(self, job_id, offset=0):

#         tasks = self.jobs[job_id]["tasks"]

#         duration = 0.0
#         for task in tasks:
#             if ("job" not in task) and ("action" not in task):
#                 d = self._get_duration_task(task)
#                 print(f"{task}: {d}s")
#                 duration += d

#         if duration >= 0.0:
#             duration -= offset
#             if duration <= 0.0:
#                 duration = 0.01

#         return duration

    def get_timestamps(self, job_id, offset=0):
        """
        Considera exatamente o tempo de execuçao do trecho pelo Lemonade.
        O calculo e feito a partir da subtraçao do maior timestamp pelo menor

        :param job_id:
        :return:
        """

        # if self.platform == 1:
        #     duration = self._get_timestamp_as_spark(job_id, offset)
        # else:
        #     duration = self._get_timestamp_as_pandas(job_id)
        
        tasks = self.jobs[job_id]["tasks"]

        duration = 0.0
        for task in tasks:
            if ("job" not in task) and ("action" not in task):
                d = self._get_duration_task(task)
                # print(f"{task}: {d}s")
                duration += d

        if duration == -1:
            # se for -1, significa que esse fluxo nao foi executado, entao devemos remover esse flow
            self.jobs.pop(job_id)
        else:
            if duration == 0.0:
                duration = 0.01
            self.jobs[job_id]["total_seconds"] = duration

    def _gen_dataflow_v5(self, base_calibration=True):
        """
        
        Ideia: `total_seconds` - (tarefas nao interessantes)
        Como o modo complete, considera todo o fluxo. 
        """

        task_orders = self.find_jobs_order()
        
        task_idx = 1
        time_idx = 2
        operations_idx = 6
        log_idx = 7

        is_not_calibration = "dc_w" not in self.workflow["name"]
        if is_not_calibration and base_calibration:
            return 
        
        
        if base_calibration:
            scenario = self.workflow["name"].split("_")[1][1:]

            # TODO: 
            if "cleaning-mode" in scenario:
                scenario = "clean-missing"
            elif "svm" in scenario:
                scenario = "svm-classification-model"
            elif "filter" in scenario:
                scenario = "filter-selection"
        else:
            scenario = self.workflow["name"]
        
        dataflow = [0 for _ in range(len(self.jobs))]

        if self.cluster:
            cluster_id = self.cluster.cluster_id
        else:
            cluster_id = -1

        #print("N jobs: ",len(task_orders))
        
        for order_job, job_id in enumerate(task_orders):
            try:
                self.jobs[job_id]["cluster"] = self.cluster
                self.jobs[job_id]["order"] = order_job
            except e:
                print(job_id)
                print(task_orders)
                

            dataflow[order_job] = [self.job_id,                             # job id
                                   "",                                      # task_id
                                   self.jobs[job_id].get("total_seconds"),  # time in seconds
                                   self.platform,                           # platform_id
                                   cluster_id,                              # cluster_id
                                   scenario,                                # scenario
                                   {},                                      # operations
                                   {"to-remove": 0, 'target-platform': ''}  # log 
                                   ]

            tasks = self.jobs[job_id]["tasks"]
            sorted_tasks = list(nx.topological_sort(self.graph.subgraph(tasks)))

            for t in tasks:
                if "action" in t:
                    self.jobs[job_id]["action"] = self.graph.nodes[t]['task_id']

            tasks_to_remove = set()
            last_slug = None
            for task_id in sorted_tasks:
                task = self.graph.nodes[task_id]
                
                slug = task.get("operation", {"slug": None})["slug"]
                dataflow[order_job][task_idx] = task_id
                
                if slug:
                    if not base_calibration:
                        scenario = slug
                        dataflow[order_job][5] = scenario
                        
                    db = None
                    try:
                        operation = copy.deepcopy(task["object"])
                    except e:
                        raise Exception(self.ERROR_SLUG_NOT_FOUND)
                        
                    # Seleção das tarefas de interesse: Tudo que estiver em
                    # tasks_to_remove, será removido do tempo de execução do job
                     
                    if (scenario in slug) or not base_calibration:
                        dataflow[order_job][operations_idx][slug] = operation
                        
                    elif (scenario == 'data-migration') and (slug == 'data-writer'):
                        p = operation.parameters
                        new_op = self.operation_api.all_operations[scenario](p)
                        dataflow[order_job][operations_idx][scenario] = new_op
                        tasks_to_remove.add(task_id)

                        dataflow[order_job][time_idx] = self.execution_time
                        
                    elif slug != 'data-writer':
                        #print(f"Removing {task_id} -> {slug}")
                        tasks_to_remove.add(task_id)

                    
                    # Operacional: avalia a movimentação dos dados entre as tarefas
                    if slug == "data-reader":
                        base_id = int(task["parameters"]['data_source'])
                        db = Dataset(base_id)
                        db_size = db.disk_size
                        if db_size < 0:
                            raise Exception(self.ERROR_UNKNOWN_SIZE.format(db.name, db.base_id))

                        new_dfs = copy.deepcopy(operation.estimate_output([db.stats]))
                        self.df_states[task_id] = new_dfs
                        self.graph.nodes[task_id]["model"] = new_dfs
                    else:
                        dfs = []

                        for input_data_order in sorted(task["parents"].keys()):
                            task_p = task["parents"][input_data_order]
                            dfs.append(self.df_states[task_p][0])
                                
                        # for p in task["parents"]:
                        #     dfs.append(self.df_states[p][0])

                        # TODO: suporte ao split
                        new_dfs = copy.deepcopy(operation.estimate_output(dfs))
                        self.graph.nodes[task_id]["model"] = new_dfs
                        self.df_states[task_id] = new_dfs
                        
                        if (scenario == 'data-migration') and (slug == 'data-writer'):
                            dataflow[order_job][operations_idx][scenario].estimate_output(dfs)

            # remove reader-data time
            for task_id in tasks_to_remove:
                time_to_remove = self._get_duration_task(task_id)
                #print(f"Minus {time_to_remove} (task: {task_id})")
                dataflow[order_job][log_idx]["to-remove"] += time_to_remove
                #dataflow[order_job][time_idx] -= time_to_remove
                dataflow[order_job][log_idx]['target-platform'] = self.graph.nodes[task_id]['attr_dict']['forms']['comment']['value']
            
            
            if dataflow[order_job][time_idx] < 0:
                dataflow[order_job][time_idx] = 0.01
                
        return dataflow
