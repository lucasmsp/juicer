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

import traceback

# !pip3 install pymysql

STAND_DB = "stand"
TAHITI_DB = "tahiti"
LIMONERO_DB = "limonero"

HOST = "150.164.203.17"
MYSQL_PORT = 43062

def get_sql_connection(db_name="stand"):
    # Connect to the database
    connection = pymysql.connect(host=HOST,
                                 user='root',
                                 password='lemon',
                                 port=MYSQL_PORT,
                                 db=db_name,
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection


def get_last_job(connection):
    with connection.cursor() as cursor:
        sql = f"SELECT MAX(id) as id from {STAND_DB}.job"
        cursor.execute(sql)
        result = cursor.fetchone()['id']
    connection.commit()
    return result


def get_job(connection, job_id):
    with connection.cursor() as cursor:
        sql = f"""
        SELECT * FROM {STAND_DB}.job where id = {job_id};
        """
        cursor.execute(sql)
        result = cursor.fetchone()
    connection.commit()
    return result


def get_action_ids(connection):
    with connection.cursor() as cursor:
        sql = f"""
        SELECT id FROM {TAHITI_DB}.operation WHERE type = "VISUALIZATION";
        """
        cursor.execute(sql)
        result = cursor.fetchall()
    connection.commit()
    action_ids = [r['id'] for r in result]

    with connection.cursor() as cursor:
        sql = """
        SELECT id FROM tahiti.operation WHERE slug LIKE "%writer%";
        """
        cursor.execute(sql)
        result = cursor.fetchall()
    connection.commit()
    action_ids += [r['id'] for r in result]
    return action_ids


def get_slug_operations(connection):
    with connection.cursor() as cursor:
        sql = f"""
        SELECT DISTINCT op.slug FROM {TAHITI_DB}.operation op WHERE op.id IN 
            (SELECT operation_id FROM {TAHITI_DB}.operation_platform WHERE platform_id IN (1, 4))
        """
        cursor.execute(sql)
        result = cursor.fetchall()
    connection.commit()
    return sorted([r['slug'] for r in result])


def get_job_log(connection, job_id):
    with connection.cursor() as cursor:
        sql = f"""
        SELECT L.date AS l_date, L.status AS l_status, S.operation_id, S.status AS s_status, S.date, S.task_id, 
        L.message, L.level, j.workflow_id
        FROM {STAND_DB}.job_step_log L 
        INNER JOIN {STAND_DB}.job_step S ON S.id = L.step_id
        INNER JOIN {STAND_DB}.job j ON j.id = S.job_id
        WHERE S.job_id = {job_id}
        """
        cursor.execute(sql)
        result = cursor.fetchall()
    connection.commit()
    return result


