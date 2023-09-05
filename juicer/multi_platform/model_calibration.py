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
from juicer.multi_platform.operations import OperationModeling
from juicer.service.tahiti_service import query_tahiti

from juicer.multi_platform.lemonade_job import LemonadeJob
from juicer.multi_platform.lemonade_job_calibration import HistoricalLemonadeJob

DEBUG = True


def log(msg):
    if DEBUG:
        print(msg)

CONFIG_FILE = "/mnt/lucasmsp/juicer/juicer-config-local.yaml"
MODELS_PATH = "/mnt/lucasmsp/thesis/models/models.pickle"

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
                    
                    if current_job_log["workflow_id"].iloc[0] == 81:
                        idx = current_job_log.index[current_job_log["message"].str.contains("Records:")][0]
                        current_job_log = current_job_log.iloc[0:idx+1]
                    
                    else:
                        # creating a task-id to the internal migration task
                        current_job_log['task_id'] = current_job_log['task_id'].mask(current_job_log["message"].str.contains("was converted"), other=current_job_log['task_id'].apply('{}-migration'.format))
                    
                    current_job_log = current_job_log[["task_id", "l_status", "operation_id", "l_date", 'message']]\
                        .groupby(['task_id', "operation_id"]).agg(MIN=("l_date", min), MAX=("l_date", max))\
                        .sort_values(["MAX", "MIN"])\
                        .reset_index()
                    
                    seconds = []
                    last = None
                    for idx, row in current_job_log.iterrows():
                        if idx  == 0:
                            last = row["MAX"]
                            d = max([(row["MAX"] - row["MIN"]).seconds, 0.01])
                            seconds.append(d)
                        else:
                            d = max([(row["MAX"] - last).seconds, 0.01])
                            seconds.append(d)
                            last = row["MAX"]

                    current_job_log["seconds"] = seconds
                    
                    current_job_log = current_job_log.to_dict('records')
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
            log(f"Starting from job: {ids[0]}")
            log(f"Last Job: {ids[-1]}")

            for current_job_id in ids:
                hlj = self.data_gathering_one(current_job_id, mode)
                if hlj:
                    self.lemonade_jobs[current_job_id] = hlj

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
   
    
    def gen_output_returns(self, lj=None):

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

        rows = []
        operations_idx = 10

        for idx, r in enumerate(tmp):
            for slug, op in r[operations_idx].items():
                try:
                    rows.append([slug, op.output])
                except Exception as e:
                    print(op.input)
                    print(op.output)
                    print(tmp[idx])
                    print(str(e))
                    traceback.print_exc()

        return rows
    
    def gen_dataflow_model_v5(self, lj=None, base_calibration=True):

        if lj:
            jobs_to_gen = {"single-input": lj}
        else:
            jobs_to_gen = self.lemonade_jobs
    
        tmp = []
        for lj in jobs_to_gen.values():
            
            try:
                jobs = lj.get_dataflow("v5", base_calibration)
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
        operations_idx = 6
        logs_idx = 7
        
        for idx, r in enumerate(tmp):
            t = [r[i] for i in range(operations_idx)] + [r[logs_idx]]
            dataflow.append(t)
            
            row = {}
            for slug, op in r[operations_idx].items():
                #try:
                f1 = op.gen_model(platform_target=1)
                f4 = op.gen_model(platform_target=4)
                #except Exception as e:
                #    print(op.input)
                #    print(op.output)
                #    print(tmp[idx])
                #    print(str(e))
                #    traceback.print_exc()
                for param, v in f1.items():
                    row["{}-spark-{}".format(slug, param)] = v
                for param, v in f4.items():
                    row["{}-pandas-{}".format(slug, param)] = v
            rows.append(row)

        rows = pd.DataFrame.from_dict(rows)
    
        dataflow = pd.DataFrame(dataflow, columns=["lemonade_id", "task_id", "total_seconds", 
                                                   "platform_id", "cluster_id", "scenario", "logs"])
        dataflow = pd.merge(dataflow, rows, how='inner', left_index=True, right_index=True)
        dataflow = dataflow.fillna(0)
        dataflow["total_seconds"] = dataflow["total_seconds"].replace(0.00, 0.001)

        return dataflow

    def gen_prediction_model(self, lj=None):

        jobs_to_gen = {"single-input": lj}

    
        tmp = []
        for lj in jobs_to_gen.values():
            try:
                jobs = lj.get_dataflow("v5")
                graph = lj.graph
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


        rows = []

        operations_idx = 6
        task_id_idx = 1
        for idx, r in enumerate(tmp):
            task_id = r[task_id_idx]
            row = {"task_id": task_id}
            for slug, op in r[operations_idx].items():
                try:
                    row["slug"] = slug
                    row["output"] = op.output
                    row['input-size'] = op.total_input_size_bytes_memory
                    row['input-rows'] = op.total_input_rows
                    f1 = op.gen_model(platform_target=1)
                    f4 = op.gen_model(platform_target=4)
                except Exception as e:
                    print(op.input)
                    print(op.output)
                    print(tmp[idx])
                    print(str(e))
                    traceback.print_exc()
                    
                for param, v in f1.items():
                    row["{}-spark-{}".format(slug, param)] = v
                for param, v in f4.items():
                    row["{}-pandas-{}".format(slug, param)] = v
            rows.append(row)

        
        return rows, graph
    
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

#def main():
#    return CostModel()

#if __name__ == "__main__":
#    main()
