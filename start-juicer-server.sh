
cd /mnt/lucasmsp/juicer;
source /mnt/lucasmsp/juicer/venv/bin/activate ;
echo $PYTHONPATH;
echo $SPARK_HOME;
echo $PATH;
SPARK_HOME=/mnt/lucasmsp/spark-3.3.1-bin-hadoop3 PYTHONPATH=/mnt/lucasmsp/juicer JUICER_CONFIG=/mnt/lucasmsp/juicer/juicer-config-local.yaml python /mnt/lucasmsp/juicer/juicer/runner/server.py -c /mnt/lucasmsp/juicer/juicer-config-local.yaml
deactivate
