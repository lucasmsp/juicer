# coding=utf-8

import pymysql.cursors
import pymysql
import json
import os
import time
import re

import numpy as np
import requests
import json
import yaml
import sys

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import datetime
import gettext
import traceback
import pickle

from juicer.multi_platform.auxiliar_services import *
from juicer.multi_platform.jobs_api import LemonadeJob, HistoricalLemonadeJob
from juicer.multi_platform.auxiliar_services import get_sql_connection
from juicer.multi_platform.operations import OperationModeling
from juicer.service.tahiti_service import query_tahiti

DEBUG = True


def log(msg):
    if DEBUG:
        print(msg)

CONFIG_FILE = "/mnt/lucasmsp/juicer/juicer-config-local.yaml"
MODELS_PATH = "/mnt/lucasmsp/models.pickle"

class CostModel(object):

    def __init__(self, current_job_id=None, last_job_id=None):
        self.lemonade_jobs = {}
        self.lemonade_jobs_excluded_by_database = []
        self.lemonade_jobs_excluded_by_operation = []
        self.lemonade_jobs_excluded_by_logs = []

        # Connect to the database
        self.connection = get_sql_connection()

        self.actions_ids = get_action_ids(self.connection)
        self.operation_slugs = get_slug_operations(self.connection)

        if not current_job_id:
            current_job_id = 0
        self.current_job_id = current_job_id

        if not last_job_id:
            last_job_id = get_last_job(self.connection)
        self.last_job_id = last_job_id

        config = CONFIG_FILE
        juicer_config = {}
        with open(config) as config_file:
            juicer_config = yaml.load(config_file.read(), Loader=yaml.FullLoader)

        locales_path = os.path.join(os.path.dirname(__file__), 'i18n', 'locales')
        t = gettext.translation('messages', locales_path, ["en"], fallback=True)
        t.install()
        self.juicer_config = juicer_config
        self.models = {}

    def get_workflow_by_job_id(self, job_id):
        """
        Usado para testes. Recupera o workflow como um dicionário 
        python a partir do id de um job no Lemonade.
        """
        current_job = get_job(self.connection, job_id)
        if current_job:
            workflow = json.loads(current_job["workflow_definition"])
            return workflow

    def get_workflow_by_id(self, workflow_id):
    
        tahiti_conf = self.juicer_config['juicer']['services']['tahiti']

        workflow = query_tahiti(base_url=tahiti_conf['url'], item_path='/workflows',
                                token=str(tahiti_conf['auth_token']),
                                item_id=workflow_id)
        return workflow


    def _collect_stand_job_log(self, current_job_id):
        current_job_log = get_job_log(self.connection, current_job_id)

        if len(current_job_log) > 0:
            not_completed = any(status != "COMPLETED"
                                for status in [step["s_status"] for step in current_job_log])

            if not not_completed:
                current_job_log = pd.DataFrame.from_dict(current_job_log)

                if len(current_job_log[
                    (current_job_log["level"] != "INFO") |
                    (current_job_log["l_status"].isin(["ERROR", "PENDING", "CANCELED"])) |
                    (current_job_log["message"].isin(["Tarefa executando (usando dados em cache)",
                                                      "Task running (cached data)"]))
                ]) == 0:

                    current_job_log = current_job_log[["task_id", "l_status", "operation_id", "l_date"]]\
                            .to_dict('records')
                    return current_job_log

        self.lemonade_jobs_excluded_by_logs.append(current_job_id)
        return None

    def data_gathering_one(self, current_job_id, mode="break-by-actions"):

        current_job = get_job(self.connection, current_job_id)
        hlj = None
        if current_job:
            if current_job["status"] == "COMPLETED":
                current_job_log = self._collect_stand_job_log(current_job_id)
                if current_job_log:
                    try:
                        hlj = HistoricalLemonadeJob(current_job, current_job_log, self.actions_ids,
                                                    self.juicer_config, self.operation_slugs, mode=mode)
                    except Exception as e:
                        msg = str(e)
                        if "[WARN] Skipping lemonade job because it contains an operation" in msg:
                            self.lemonade_jobs_excluded_by_operation.append(current_job_id)
                        elif "[WARN] Skipping lemonade job because one of it's input data size is unknown" in msg:
                            self.lemonade_jobs_excluded_by_database.append(current_job_id)
                        else:
                            print(str(e) + " - Lemonade_id: {}".format(current_job_id))
                            traceback.print_exc()

        return hlj

    def data_gathering_multiple(self, ids=None, mode="break-by-actions"):
        if ids:
            ids = sorted(ids)
            current_job_id = ids[0]
            last_job_id = ids[-1]
        else:
            current_job_id = self.current_job_id
            last_job_id = self.last_job_id

        log(f"Starting from job: {current_job_id}")
        log(f"Last Job: {last_job_id}")

        while current_job_id <= last_job_id:
            hlj = self.data_gathering_one(current_job_id, mode)
            if hlj:
                self.lemonade_jobs[current_job_id] = hlj
            current_job_id += 1

    def gen_dataflow_model_v5(self, lj=None):

        if lj:
            jobs_to_gen = {"single-input": lj}
        else:
            jobs_to_gen = self.lemonade_jobs
    
        tmp = []
        for lj in jobs_to_gen.values():
            try:
                jobs = lj.get_dataflow("v5")
                if jobs:
                    for d in jobs:
                        tmp.append(d)
            except Exception as e:
                msg = str(e)
                if "[WARN] Skipping lemonade job because it contains an operation" in msg:
                    self.lemonade_jobs_excluded_by_operation.append(lj.job_id)
                elif "[WARN] Skipping lemonade job because one of it's input data size is unknown" in msg:
                    self.lemonade_jobs_excluded_by_database.append(lj.job_id)
                else:
                    print("gen_dataflow_model_v5: " + str(e) + " - Lemonade_id: {}".format(lj.job_id))
                    print(traceback.format_exc())

        dataflow = []
        rows = []
        operations_idx = 8

        for r in tmp:
            t = [r[i] for i in range(operations_idx)]
            dataflow.append(t)
            row = {}
            for slug, op in r[operations_idx].items():
                f = op.gen_model()
                for param, v in f.items():
                    row["{}-{}".format(slug, param)] = v
            rows.append(row)

        rows = pd.DataFrame.from_dict(rows)
        dataflow = pd.DataFrame(dataflow, columns=["lemonade_id", "job_order", "job_parents",
                                                    "total_seconds", "platform_id",
                                                    "n_cores", "memory_ram", "scenario"])
        dataflow = pd.merge(dataflow, rows, how='inner', left_index=True, right_index=True)
        dataflow = dataflow.fillna(0)
        dataflow["total_seconds"] = dataflow["total_seconds"].replace(0.00, 0.001)

        return dataflow

    def load_models(self):
        """
        Carrega os modelos de regressão para cada caixa/plataforma suportada pelo Lemonade.
        """
        self.models = pickle.load(open(MODELS_PATH, 'rb'))

    # def save_database(self):
    #     filename = "/home/lucasmsp/workspace/doutorado/thesis/database.pickle"
    #     with open(filename, 'wb') as handle:
    #         pickle.dump(self.lemonade_jobs, handle)

    # def load_database(self):
    #     filename = "/home/lucasmsp/workspace/doutorado/thesis/database.pickle"
    #     self.lemonade_jobs = pickle.load(open(filename, 'rb'))

    def extract_jobs(self, workflow, mode="break-by-tasks"):
        """
        A partir de um workflow (dicionário/JSON), gera o objeto LemonadeJob
        """
        if isinstance(workflow, int):
            workflow = self.get_workflow_by_id(workflow)
        lj = LemonadeJob(workflow, self.actions_ids, self.juicer_config, self.operation_slugs, mode)
        return lj

    def where_to_run(self, rows):
        self.load_models()
        table = [{} for _ in range(len(rows))]
        for m in self.models:
            mm = self.models[m]
            for i, r in enumerate(rows):
                table[i][m] = mm.predict([r])

        result = {"predictions": table}
        duration = 0
        flow = []

        for t in table:
            spark = t["Spark"]
            pandas = t["Pandas"]
            if spark > pandas:
                flow.append("Pandas")
                duration += pandas[0]
            else:
                flow.append("Spark")
                duration += spark[0]

        result["flow"] = flow
        result["duration"] = duration
        print("where_to_run: ", result)
        return result

#     def gen_test_dataflow(self, mode="break-by-stages"):
#         current_job_id = 0
#         lemonade_jobs = []
#         ids = []
#         while current_job_id <= self.last_job_id:

#             current_job = get_job(self.connection, current_job_id)
#             if current_job_id % 1000 == 0:
#                 print("It: ", current_job_id)

#             if current_job:
#                 lj = None
#                 try:
#                     workflow = json.loads(current_job["workflow_definition"])
#                     lj = LemonadeJob(workflow, self.actions_ids,
#                                      self.juicer_config, self.operation_slugs, mode=mode)
#                     lemonade_jobs.append(lj)
#                     ids.append(current_job_id)
#                 except Exception as e:
#                     pass
#                     #print(traceback.format_exc())
#                     #print(str(e) + "- Lemonade_id: {}".format(current_job_id))
#             current_job_id += 1

#         print("Terminou de carregar os workflows!")

#         tmp = []
#         for lj, idx in zip(lemonade_jobs, ids):
#             try:
#                 for d in lj.get_dataflow("v4"):
#                     tmp.append([idx] + d)
#             except Exception as e:
#                 print(str(e))

#         print("Terminou de get_datafflow!")
#         dataflow = []
#         rows = []
#         operations_idx = 9+1
#         opmodeling = OperationModeling()
#         for r in tmp:
#             dataflow.append([r[i] for i in range(operations_idx)])
#             for key in list(r[operations_idx].keys()):
#                 ml_operation = r[operations_idx][key]
#                 if isinstance(ml_operation, dict):

#                     del r[operations_idx][key]
#                     for key2 in ml_operation:
#                         r[operations_idx]["{}_{}".format(opmodeling.clean_slug(key), key2)] = ml_operation[key2]

#             rows.append(r[operations_idx])

#         rows = pd.DataFrame.from_dict(rows)
#         dataflow = pd.DataFrame(dataflow, columns=["lemonade_id", "test_id", "job_order", "job_parents",
#                                                    "total_seconds", "platform_id", "n_cores", "memory_ram",
#                                                    "input_size", "expected_output"])
#         dataflow = pd.merge(dataflow, rows, how='inner', left_index=True, right_index=True)
#         dataflow = dataflow.fillna(0)
#         dataflow = dataflow.loc[dataflow["not-supported"] < 1.0]
#         dataflow = dataflow.drop(["data_amplification", "not-supported"], axis=1)

#         return dataflow



#def main():
#    return CostModel()

#if __name__ == "__main__":
#    main()
