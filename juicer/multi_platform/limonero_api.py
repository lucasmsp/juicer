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
import traceback
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
from bisect import bisect

# LIBRARY_PATH = os.path.expanduser("/home/lucasmsp/workspace/bigsea/docker-lemonade/juicer/")
# sys.path.insert(0, LIBRARY_PATH)
# os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + LIBRARY_PATH

import findspark
findspark.init()

from juicer.spark.data_operation import DataReaderOperation
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import QuantileDiscretizer

from juicer.multi_platform.auxiliar_services import LIMONERO_DB

#HOST = "localhost"
HOST = "150.164.203.17"
LIMONERO_PORT = 43402
# CONFIG_FILE = "/home/lucasmsp/workspace/bigsea/docker-lemonade/config/juicer-config-local.yaml"
CONFIG_FILE =  "/mnt/lucasmsp/juicer/juicer-config-local.yaml"


class Dataset(object):

    def __init__(self, base_id):
        self.disk_size = 0.00
        self.storage_name = None
        self.format = None
        self.storage_type = None
        self.url = None
        self.storage_id = None
        self.name = None
        self.base_id = base_id
        self.stats = None
        self.get_base_stats()

    def __repr__(self):
        return "Dataset(id={}, name='{}')".format(self.base_id, self.name)

    def get_base_stats(self):
        # connection = get_sql_connection(LIMONERO_DB)
        # with connection.cursor() as cursor:
        #     sql = """
        #     SELECT ds.id, ds.name, ds.url, format, estimated_size_in_mega_bytes,
        #            storage_id,  s.name AS storage_name, s.type AS storage_type
        #     FROM {LIMONERO_DB}.data_source ds
        #     INNER JOIN {LIMONERO_DB}.storage s ON s.id = ds.storage_id
        #     WHERE ds.id = {BASE_ID};
        #     """.format(LIMONERO_DB=LIMONERO_DB, BASE_ID=self.base_id)
        #     cursor.execute(sql)
        #     result = cursor.fetchone()
        # connection.commit()
        # connection.close()

        result = self.query_limonero(f"http://{HOST}:{LIMONERO_PORT}", 'datasources', '123456', self.base_id)
        if result:
            self.name = result['name']
            self.url = result['url']
            self.format = result['format']

            try:
                self.disk_size = float(result.get("estimated_size_in_mega_bytes", -1))
            except Exception:
                self.disk_size = -1

            if self.disk_size == 0.00:
                self.disk_size = 0.01

            self.storage_id = result["storage"]['id']
            self.storage_name = result["storage"]['name']
            self.storage_type = result["storage"]['type']
            self.stats = BaseStatistics(n_rows=result["estimated_rows"])
            self.stats.size_bytes_disk = self.disk_size

            for column in result["attributes"]:
                self.stats.create_new_column(**column)

    def print_stats(self):
        print(f"""
        Name: {self.name}
        url: {self.url}
        format: {self.format}
        size_in_disk: {self.disk_size}
        storage_name: {self.storage_name}
        storage_type: {self.storage_type}
        """)

    @staticmethod
    def remove_initial_final_path_separator(path):
        if path.endswith('/'):
            path = path[:-1]
        if path.startswith('/'):
            path = path[1:]
        return path

    def query_limonero(self, base_url, item_path, token, item_id):
        headers = {'X-Auth-Token': token}
        base_url = self.remove_initial_final_path_separator(base_url)
        item_path = self.remove_initial_final_path_separator(item_path)
        item_id = self.remove_initial_final_path_separator(str(item_id))

        if base_url.endswith('/'):
            base_url = base_url[:-1]

        if item_path.endswith('/'):
            item_path = item_path[:-1]

        if item_path:
            url = '{}/{}/{}'.format(base_url, item_path, item_id)
        else:
            url = '{}/{}'.format(base_url, item_id)

        # log.debug(gettext('Querying Limonero URL: %s'), url)

        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return json.loads(r.text)
        else:
            log.error(gettext('Error querying Limonero URL: %s (%s: %s)'), url,
                      r.status_code, r.text)
            if r.status_code == 404:
                msg = gettext("not found")
            else:
                msg = r.text
            raise ValueError(gettext(
                "Error loading {} id {}: HTTP {} - {} ({})").format(
                item_path, item_id, r.status_code, msg, url))


class BaseStatistics(object):

    def __repr__(self):
        return "Statistics(rowCount={}, sizeInBytesMemory={}, nColumns={})"\
            .format(self.n_rows, self.size_bytes_memory, self.n_columns)

    def __init__(self, n_rows):
        self.size_bytes_disk = None
        self.size_bytes_memory = None
        self.n_columns = None
        self.columns = None
        self.n_rows = n_rows
        self.columns = {}
        self.cols_as_features = {}

    def create_new_column(self, **values):
        if "n_rows" not in values:
            values["n_rows"] = self.n_rows
        self.columns[values["name"]] = LemonadeColumn(**values)
        self.n_columns = len(self.columns)
        self.size_bytes_memory = sum([c.estimated_column_size_total for c in self.columns.values()])

    def update_column(self, column, values):
        for k, v in values.items():
            self.columns[column].__dict__[k] = v
            # print("{} {} {}".format(column, k, v))

    def recalculate(self):
        self.n_columns = len(self.columns)
        for column in self.columns:
            self.columns[column].estimate_size()
        self.size_bytes_memory = sum([c.estimated_column_size_total for c in self.columns.values()])
        self.n_rows = max([self.columns[c].n_rows for c in self.columns])
    
    def remove_n_rows(self, rows_to_remove):
        if rows_to_remove < self.n_rows:
            for column in self.columns:
                self.columns[column].remove_n_rows(rows_to_remove)
            self.recalculate()
        


class LemonadeColumn(object):
    # https://github.com/apache/spark/blob/master/sql/catalyst/src/main/scala/org/apache/spark/sql/types/
    # https://github.com/apache/spark/blob/master/core/src/main/scala/org/apache/spark/util/SizeEstimator.scala

    LIMONERO_DATA_TYPES_TO_BYTES = {
        "BINARY": 1,
        "CHARACTER": 1,
        "DATETIME": 8,
        "DATE": 4,
        "DOUBLE": 8,
        "DECIMAL": 8,  # TODO: 8 bytes when precision is at most 18 and 16 bytes otherwise
        "FLOAT": 4,
        "LONG": 8,
        "INTEGER": 4,
        "TEXT": 1
    }

    def __init__(self, **column):

        self.n_rows = None
        self.name = None
        self.type = None
        self.estimated_column_size = 0
        self.missing_total = 0
        self.size = None
        self.min_value = None
        self.max_value = None
        self.distinct_values = None
        self.deciles = None

        self.estimated_column_size_total = 0
        self.__dict__.update(column)
        self.estimate_size()
        
        if not self.distinct_values:
            self.distinct_values = self.n_rows
    
        if self.deciles:
            #print(self.deciles)
            self.deciles = self.deciles.replace('"', "'").replace("{'", '{"').replace("': ", '": ').replace(", '", ', "').replace("\'", '\\"')
            self.deciles = json.loads(self.deciles)
            new_deciles = {}

            if self.type in ["DOUBLE", "DECIMAL"]:
                for k,v in self.deciles.items():
                    new_deciles[float(k)] = v
            elif self.type in ["INTEGER", "LONG"]:
                for k,v in self.deciles.items():
                    new_deciles[int(k.split(".")[0])] = v
            else:
                for k,v in self.deciles.items():
                    new_deciles[k] = v

            self.deciles = new_deciles
                

    def estimate_size(self):
        # https://github.com/apache/spark/blob/master/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/plans/logical/statsEstimation/EstimationUtils.scala
        self.estimated_column_size = self.LIMONERO_DATA_TYPES_TO_BYTES[self.type]
        if self.missing_total:
            self.missing_total = int(self.missing_total)
        else:
            self.missing_total = 0

        if self.type in ["TEXT", "CHARACTER", "BINARY"]:
            if not self.size:
                # Spark's default value
                self.size = 20
            self.estimated_column_size = self.LIMONERO_DATA_TYPES_TO_BYTES[self.type] * self.size
            if self.type in ["TEXT", "CHARACTER"]:
                self.estimated_column_size += (4+8)

        self.estimated_column_size_total = self.estimated_column_size * self.n_rows - \
            (self.LIMONERO_DATA_TYPES_TO_BYTES[self.type] * self.missing_total)

    def __repr__(self):
        return "name: '{}', type: '{}', estimated_column_size_total: {}, n_rows: {}\n"\
            .format(self.name, self.type, self.estimated_column_size_total, self.n_rows)
    
    def is_number(self):
        return self.type in ["BINARY", "DOUBLE", "DECIMAL", "FLOAT", "LONG", "INTEGER"]
    
    def is_string(self):
        return self.type in ["TEXT", "CHARACTER"]
    
    def remove_n_rows(self, rows_to_remove):
        r1 = rows_to_remove // 2
        
        if self.missing_total < r1:
            r1 = self.missing_total
            self.missing_total = 0
        else:
            self.missing_total -= r1
            
        r2 = rows_to_remove - r1
        
        self.n_rows -= r2

    def get_bin_to_value(self, v1, mode="equals"):
        if self.deciles:
            deciles_keys = sorted(list(self.deciles.keys()))
            greater = []
            equals = []
            lesser = []

            for i, v in enumerate(deciles_keys):
                if v >= v1:
                    greater.append(i)
                    if len(equals) == 0:
                        equals.append(i)
                else:
                    lesser.append(i)

            if mode == "equals":
                return equals
            elif mode == "greater":
                return greater
            else:
                return lesser
        else:
            return None
        
    def get_elements_interval(self, overlap_interval):
        if not overlap_interval:
            return 0
        
        elif self.deciles:
            tmp = {}
            # filtrando elementos menores que o máximo
            for k in sorted(list(self.deciles.keys()), reverse=True):
                if k < overlap_interval[1]:
                    tmp[k] = self.deciles[k]
            idx = 0
            for i, k in enumerate(sorted(list(tmp.keys()))):
                if k > overlap_interval[0]:
                    idx = i
                    break
        
            for i, k in enumerate(sorted(list(tmp.keys()))):
                if i < idx:
                    del tmp[k]

            value = sum([tmp[k] for k in tmp])
            return value
        else:
            return self.n_rows

            
                


class LimoneroCalibration(object):

    def __init__(self):

        self.last_datasource_id = None
        self.current_datasource_id = 1
        self.load_id()

    def save_id(self, datasource_id):
        with open("/tmp/limonero_info.txt", "w") as f:
            f.write(f"{datasource_id}")

    def load_id(self):
        if os.path.exists("/tmp/limonero_info.txt"):
            with open("/tmp/limonero_info.txt", "r") as f:
                self.current_datasource_id = float(f.read())

    def get_last_datasource_id(self):
        connection = get_sql_connection(LIMONERO_DB)
        with connection.cursor() as cursor:
            sql = f"SELECT MAX(id) as id from {LIMONERO_DB}.data_source"
            cursor.execute(sql)
            result = cursor.fetchone()['id']
        connection.commit()
        connection.close()
        self.last_datasource_id = result
        return result

    def check_datasource_id(self, data_source_id):
        connection = get_sql_connection(LIMONERO_DB)
        with connection.cursor() as cursor:
            sql = f"SELECT id as id from {LIMONERO_DB}.data_source WHERE id = {data_source_id}"
            cursor.execute(sql)
            result = cursor.fetchone()
        connection.commit()
        connection.close()
        has_id = True if result else False
        return has_id

    def update_limonero(self, sqls):
        connection = get_sql_connection(LIMONERO_DB)
        for sql in sqls:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                cursor.fetchall()
            connection.commit()
        connection.close()

    def calibrate(self, datasource_id=None):

        if not datasource_id:
            self.get_last_datasource_id()
            print("Current ID: {} | Last ID: {}".format(self.current_datasource_id, self.last_datasource_id))
            self.last_datasource_id +=1
        else:
            self.last_datasource_id = datasource_id+1
            self.current_datasource_id = datasource_id

        if self.last_datasource_id == self.current_datasource_id:
            print("Limonero info is already calibrated ...")
            return

        juicer_config = yaml.load(
            open(CONFIG_FILE).read(),
            Loader=yaml.FullLoader)

        app_configs = juicer_config['juicer']['spark']

        spark_builder = SparkSession.builder.appName('Helper')
        for option, value in app_configs.items():
            spark_builder = spark_builder.config(option, value)
        spark_session = spark_builder.getOrCreate()

        datasources = [i for i in range(self.current_datasource_id, self.last_datasource_id)]
        stats = {}
        for datasource_id in datasources:
            print(f"Checking datasource {datasource_id} ...")
            if not self.check_datasource_id(datasource_id):
                print(f"Datasource {datasource_id} does not exist.")
                continue

            try:
                parameters = {
                    "data_source": datasource_id,
                    "mode": "PERMISSIVE",
                    'configuration': {
                        'juicer': {
                            'services': {
                                'limonero': {
                                    'url': f'http://{HOST}:{LIMONERO_PORT}',
                                    'auth_token': '123456'
                                }
                            }
                        }
                    },
                    'workflow': {'data_source_cache': {}}
                }
                n_out = {'output data': 'df'}
                data_reader = DataReaderOperation(parameters, named_inputs={}, named_outputs=n_out)
                code = "from pyspark.sql import types\n"+data_reader.generate_code()
                code = code[:code.rfind('\n')]
                _locals = locals()
                exec(code, globals(), _locals)
                df = _locals["df"]
                estimated_rows = df.count()
                stats[datasource_id] = {"estimated_rows": estimated_rows, "attribute": {}}
                distinct_values = df.agg(*(F.countDistinct(F.col(c)).alias(c) for c in df.columns)) \
                    .toPandas() \
                    .to_dict(orient="records")[0]

                # Computes specified statistics for numeric and string columns.
                summary = df.summary().toPandas().set_index("summary").T.to_dict(orient='index')
                dtypes = {k: v for k, v in df.dtypes}

                for c in df.columns:
                    col_type = dtypes[c]
                    
                    missing_total = 0
                    mean = "NULL"
                    std_deviation = "NULL"
                    deciles = "NULL"
                    median = 'NULL'
                    min_value = "NULL"
                    max_value = "NULL"
                    
                    
                    if c in summary:
                        # numeric and string columns
                        min_value = summary[c]['min']
                        max_value = summary[c]['max']
                        missing_total = estimated_rows - int(summary[c]['count'])
                    else:
                        # other dtypes, such as datetime
                        result = df.agg(F.min(c).alias("min"), F.max(c).alias("max"), F.count(c).alias("count")).limit(1).collect()[0]
                        min_value = result.__getitem__('min')
                        max_value = result.__getitem__('max')
                        missing_total = estimated_rows - result.__getitem__('count')


                    if (col_type not in ["string"]) and (c in summary):
                        # only numeric fields
                        median = summary[c]['50%']

                        if summary[c]['mean'] not in [None, "NaN", "Infinity"]:
                            mean = float(summary[c]['mean'])
                            if mean >= 2e30:
                                mean = "NULL"

                        if summary[c]['stddev'] not in ["NaN", None]:
                            std_deviation = float(summary[c]['stddev'])
                            if std_deviation >= 2e30:
                                std_deviation = "NULL"

                        # the following part will be executed only to numerical data
                        qds1 = QuantileDiscretizer(inputCol=c, outputCol="buckets")  # [lower, upper)
                        qds1.setNumBuckets(10)
                        bucketizer = qds1.fit(df)
                        bucket_range = bucketizer.getSplits()

                        counts = bucketizer.transform(df).groupby("buckets")\
                            .agg(F.count(F.col(c)).alias("count"))\
                            .orderBy("buckets").collect()

                        bucket_range[-1] = max_value

                        freq = {int(counts[i].__getitem__('buckets')): counts[i].__getitem__('count')
                                for i in range(len(counts)) if counts[i].__getitem__('buckets')}

                        deciles = {bucket_range[i + 1]: freq.get(i) for i in freq}
                        deciles = "'"+json.dumps(deciles).replace('"', '\"')+"'"
    
                        missing_total = estimated_rows - int(summary[c]['count'])
                    
                    elif col_type in ["string"]:
                        top10 = df.select(c).groupby(c).count()\
                            .orderBy(F.col(c).desc())\
                            .limit(10)\
                            .collect()
                        
                        deciles = {top10[i].__getitem__(c): top10[i].__getitem__('count')
                                for i in range(len(top10))}
                        deciles = "'"+json.dumps(deciles).replace('"', '\"')+"'"
                    elif col_type in ["datetime"]:
                        #!TODO
                        pass

                    stats[datasource_id]['attribute'][c] = {'distinct_values': distinct_values[c],
                                                            'missing_total': missing_total,
                                                            'mean': mean,
                                                            'min': min_value,
                                                            'max': max_value,
                                                            'std_deviation': std_deviation,
                                                            'median': median,
                                                            'deciles': deciles}
            except Exception as err:
                print(f"Unable to process datasource {datasource_id}")
                print(err)
                print(traceback.format_exc())

            sqls = ["UPDATE {}.data_source SET estimated_rows={} WHERE id = {}"
                    .format(LIMONERO_DB, stats[row]['estimated_rows'], row) for row in stats]
            self.update_limonero(sqls)

            sqls = ["""
            UPDATE {LIMONERO_DB}.attribute 
            SET min_value='{min_value}',
                max_value='{max_value}', 
                mean_value={mean_value}, 
                std_deviation={std_deviation},
                median_value={median_value}, 
                distinct_values={distinct_values}, 
                missing_total={missing_total},
                deciles={deciles}
            WHERE name='{name}' AND data_source_id = {data_source_id}; 
            """.format(LIMONERO_DB=LIMONERO_DB, min_value=stats[ds]['attribute'][att]["min"],
                       max_value=stats[ds]['attribute'][att]["max"],
                       mean_value=stats[ds]['attribute'][att]["mean"],
                       median_value=stats[ds]['attribute'][att]["median"],
                       std_deviation=stats[ds]['attribute'][att]["std_deviation"],
                       distinct_values=stats[ds]['attribute'][att]["distinct_values"],
                       missing_total=stats[ds]['attribute'][att]["missing_total"],
                       deciles=stats[ds]['attribute'][att]["deciles"],
                       name=att,
                       data_source_id=ds) for ds in stats for att in stats[ds]["attribute"]]

            self.update_limonero(sqls)
            self.save_id(self.last_datasource_id)
            spark_session.stop()
        print("Limonero's calibration is finished ...")


if __name__ == "__main__":
    ds = LimoneroCalibration()
    ds.calibrate()

