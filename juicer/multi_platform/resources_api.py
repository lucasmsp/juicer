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

from juicer.multi_platform.auxiliar_services import get_sql_connection
from juicer.multi_platform.auxiliar_services import STAND_DB


class Cluster(object):

    def __init__(self, cluster_id):
        self.executors = None
        self.executor_memory = None
        self.executor_cores = None
        self.address = None
        self.enabled = None
        self.description = None
        self.general_parameters = None
        self.name = None
        self.cluster_id = cluster_id
        self.get_cluster_conf(cluster_id)

    def get_cluster_conf(self, cluster_id):
        connection = get_sql_connection(STAND_DB)
        with connection.cursor() as cursor:
            sql = """
            SELECT * FROM {STAND_DB}.cluster where id = {CLUSTER_ID};
            """.format(STAND_DB=STAND_DB, CLUSTER_ID=cluster_id)
            cursor.execute(sql)
            result = cursor.fetchone()
        connection.commit()
        connection.close()

        self.name = result['name']
        self.description = result['description']
        self.enabled = result['enabled'] == '1'
        self.address = result['address']
        self.executor_cores = int(result['executor_cores'])
        self.executor_memory = result['executor_memory'].upper()
        if "GB" in self.executor_memory:
            self.executor_memory = float(self.executor_memory.replace("GB", ""))
        elif "G" in self.executor_memory:
            self.executor_memory = float(self.executor_memory.replace("G", ""))
        else:
            raise
        self.executors = int(result['executors'])
        self.general_parameters = result.get('general_parameters', "")
        if self.general_parameters:
            self.general_parameters = self.general_parameters.split(",")

    def print_conf(self):
        print(f"""
        Name: {self.name}
        Address: {self.address}
        Executor_cores: {self.executor_cores}
        Executors: {self.executors}
        Executor_memory: {self.executor_memory}
        General_parameters: {self.general_parameters}
        """)
