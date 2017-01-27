# coding=utf-8
from __future__ import print_function

import json
from textwrap import dedent

from datetime import datetime

import mock
import os
from dummy_spark import SparkConf, SparkContext
from juicer.runner.control import StateControlRedis
from juicer.spark.spark_minion import SparkMinion
from mockredis import mock_strict_redis_client

config = {
    'juicer': {
        'servers': {
            'redis_url': 'redis://invalid:2923'
        }
    }
}


# Auxiliary functions
def get_records():
    return [
        ['Brazil', 'UFMG', 223],
        ['Spain', 'UPV', 52],
        ['Italy', 'POLIMI', 921],
    ]


def get_side_effect(records, task_id, index=0):
    # noinspection PyUnusedLocal
    def side_effect0(w, g, c, out):
        print(dedent("""
            from dummy_spark import SparkConf, SparkContext
            import datetime
            sconf = SparkConf()
            sc = SparkContext(master='', conf=sconf)
            records = {records}
            rdd = sc.parallelize(records)
            class DataFrame:
                def __init__(self, rdd):
                    self.rdd = rdd

                def take(self, total):
                    return self
            def main(spark_session, cached_data):
                return {{
                    '{task_id}': (DataFrame(rdd), 20.0)
                }}
            """.format(records=json.dumps(records), task_id=task_id)), file=out)

    # noinspection PyUnusedLocal
    def side_effect1(w, g, c, out):
        print(dedent("""
            def main(spark_session, cached_data):
                return {
                    "xyz-647": ("df", 27.27)
                }"""), file=out)

    # noinspection PyUnusedLocal
    def side_effect2(w, g, c, out):
        print(dedent("""
            def main(spark_session, cached_data):
                return {'res': 'version 1.0'} """), file=out)

    def side_effect3(w, g, c, out):
        print(dedent("""
            def main(spark_session, cached_data):
                return {'res': 'version 2.1'} """), file=out)

    # noinspection PyUnusedLocal
    def side_effect4(w, g, c, out):
        print(dedent("""
            a = 4
            def main(spark_session, cached_data):
                return Invalid Code """), file=out)

    return \
        [side_effect0, side_effect1, side_effect2, side_effect3, side_effect4][
            index]


class DataFrame:
    def __init__(self, rdd):
        self.rdd = rdd

    def take(self, total):
        return self


# noinspection PyProtectedMember
def test_minion_ping_success():
    workflow_id = 6666
    app_id = 897987

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        redis_conn = mocked_redis()
        minion = SparkMinion(redis_conn=redis_conn,
                workflow_id=workflow_id, app_id=app_id,
                config=config)
        minion._perform_ping()

        state_control = StateControlRedis(redis_conn)

        assert json.loads(state_control.get_minion_status(app_id)) == {
            'status': 'READY', 'pid': os.getpid()}
        assert state_control.get_app_output_queue_size(app_id) == 0


# noinspection PyProtectedMember
def test_minion_generate_output_success():
    workflow_id = 6666
    app_id = 897987

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        redis_conn = mocked_redis()
        minion = SparkMinion(redis_conn=redis_conn,
                workflow_id=workflow_id, app_id=app_id,
                config=config)

        state_control = StateControlRedis(redis_conn)

        msgs = ["Message being sent \n{}", {'msg': 'Dictionary being sent'}]
        for msg in msgs:
            minion._generate_output(msg)
            result = json.loads(
                state_control.pop_app_output_queue(app_id, False))
            assert sorted(result.keys()) == sorted(
                ['date', 'status', 'message', 'workflow_id', 'app_id', 'code'])
            assert result['app_id'] == app_id
            assert result['message'] == msg
        assert state_control.get_app_output_queue_size(app_id) == 0


# noinspection PyProtectedMember
def test_minion_perform_execute_success():
    workflow_id = '6666'
    app_id = '897447'
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = get_side_effect(None, 0, 1)

            redis_conn = mocked_redis()
            minion = SparkMinion(redis_conn=redis_conn,
                    workflow_id=workflow_id, app_id=app_id,
                    config=config)

            # Configure mocked redis
            state_control = StateControlRedis(redis_conn)
            with open(os.path.join(os.path.dirname(__file__),
                                   'fixtures/simple_workflow.json')) as f:
                data = json.loads(f.read())

            msg = {
                'workflow_id': workflow_id,
                'app_id': app_id,
                'type': 'execute',
                'workflow': data
            }

            state_control.push_app_queue(app_id, json.dumps(msg))

            minion._process_message()

            assert minion.state == {"xyz-647": ("df", 27.27)}, 'Invalid state'

            assert state_control.get_app_output_queue_size(
                app_id) == 1, 'Wrong number of output messages'


# noinspection PyProtectedMember
def test_minion_perform_execute_reload_code_success():
    workflow_id = '6666'
    app_id = '667788'
    workflow = {
        'workflow_id': workflow_id,
        'app_id': app_id,
        'type': 'execute',
        'workflow': ''
    }
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = get_side_effect(None, None, 2)

            redis_conn = mocked_redis()
            minion = SparkMinion(redis_conn=redis_conn,
                    workflow_id=workflow_id, app_id=app_id,
                    config=config)

            # Configure mocked redis
            state_control = StateControlRedis(redis_conn)
            with open(os.path.join(os.path.dirname(__file__),
                                   'fixtures/simple_workflow.json')) as f:
                data = json.loads(f.read())
                workflow['workflow'] = data

            state_control.push_app_queue(app_id, json.dumps(workflow))

            minion._process_message()
            assert minion.state == {'res': 'version 1.0'}, 'Invalid state'

            # Executes the same workflow, but code should be different
            state_control.push_app_queue(app_id, json.dumps(workflow))
            assert state_control.get_app_output_queue_size(
                app_id) == 1, 'Wrong number of output messages'

            minion.transpiler.transpile = get_side_effect(None, None, 3)
            minion._process_message()

            assert minion.state == {'res': 'version 2.1'}, 'Invalid state'

            assert state_control.get_app_output_queue_size(
                app_id) == 2, 'Wrong number of output messages'


# noinspection PyProtectedMember
def test_minion_generate_invalid_code_failure():
    workflow_id = '6666'
    app_id = '667788'
    workflow = {
        'workflow_id': workflow_id,
        'app_id': app_id,
        'type': 'execute',
        'workflow': ''
    }
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = get_side_effect(None, None, 4)
            redis_conn = mocked_redis()
            minion = SparkMinion(redis_conn=redis_conn,
                    workflow_id=workflow_id, app_id=app_id,
                    config=config)
            # Configure mocked redis
            with open(os.path.join(os.path.dirname(__file__),
                                   'fixtures/simple_workflow.json')) as f:
                data = json.loads(f.read())
                workflow['workflow'] = data

            state_control = StateControlRedis(redis_conn)
            state_control.push_app_queue(app_id, json.dumps(workflow))
            minion._process_message()

            assert state_control.get_app_output_queue_size(
                app_id) == 2, 'Wrong number of output messages'
            # Discards
            state_control.pop_app_output_queue(app_id)

            msg = json.loads(state_control.pop_app_output_queue(app_id))
            assert msg['status'] == 'ERROR'
            assert msg['message'][:19] == 'Invalid Python code'

# noinspection PyProtectedMember
def test_minion_perform_deliver_success():
    workflow_id = 6666
    app_id = 1000
    out_queue = 'queue_2000'
    sconf = SparkConf()
    sc = SparkContext(master='', conf=sconf)

    rdd = sc.parallelize(get_records())

    df0 = DataFrame(rdd=rdd)
    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        redis_conn = mocked_redis()
        state_control = StateControlRedis(redis_conn)

        data = {
            'workflow_id': workflow_id,
            'app_id': app_id,
            'type': 'deliver',
            'task_id': '033f-284ab-28987e',
            'port': '0',
            'output': out_queue,
            'workflow': ''
        }
        state_control.push_app_queue(app_id, json.dumps(data))
        minion = SparkMinion(redis_conn=redis_conn,
                workflow_id=workflow_id, app_id=app_id,
                config=config)
        minion.state = {
            data['task_id']: (df0, 35.92)
        }
        minion._process_message()
        
        # Discard first status message
        state_control.pop_app_output_queue(app_id, False)

        msg = json.loads(state_control.pop_app_output_queue(app_id, False))
        assert msg['status'] == 'SUCCESS', 'Invalid status'
        assert msg['code'] == SparkMinion.MNN002[0], 'Invalid code'

        # CSV data
        csv_records = '\n'.join(map(minion._convert_to_csv, get_records()))

        result = state_control.pop_queue(out_queue, False)
        assert result == csv_records, 'Wrong CSV generated'


# noinspection PyProtectedMember
def test_minion_perform_deliver_missing_state_process_app_with_success():
    workflow_id = '6666'
    app_id = '1000'
    out_queue = 'queue_2000'
    task_id = '033f-284ab-28987e'
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = get_side_effect(get_records(),
                                                           task_id)
            redis_conn = mocked_redis()
            state_control = StateControlRedis(redis_conn)

            data = {
                'workflow_id': workflow_id,
                'app_id': app_id,
                'type': 'deliver',
                'task_id': task_id,
                'port': '0',
                'output': out_queue,
                'workflow': {"tasks": [], "flows":[]}
            }

            state_control.push_app_queue(app_id, json.dumps(data))
            minion = SparkMinion(redis_conn=redis_conn,
                    workflow_id=workflow_id, app_id=app_id,
                    config=config)
            minion.state = {
            }
            minion._process_message()
           
            # Discard first status message
            state_control.pop_app_output_queue(app_id, False)

            msg = json.loads(state_control.pop_app_output_queue(app_id, False))
            assert msg['status'] == 'WARNING', 'Invalid status'
            assert msg['code'] == SparkMinion.MNN003[0], 'Invalid code'

            # CSV data
            csv_records = '\n'.join(map(minion._convert_to_csv, get_records()))

            result = state_control.pop_queue(out_queue, False)
            assert result == csv_records, 'Wrong CSV generated'


# noinspection PyProtectedMember
def test_minion_perform_deliver_missing_state_process_app_with_failure():
    workflow_id = '6666'
    app_id = '6000'
    out_queue = 'queue_2000'
    task_id = '033f-284ab-28987e'
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            # Invalid code
            mocked_transpile.side_effect = get_side_effect(get_records(),
                                                           task_id, 4)
            redis_conn = mocked_redis()
            state_control = StateControlRedis(redis_conn)

            data = {
                'workflow_id': workflow_id,
                'app_id': app_id,
                'type': 'deliver',
                'task_id': task_id,
                'port': '0',
                'output': out_queue,
                'workflow': {"tasks": [], "flows":[]}
            }

            state_control.push_app_queue(app_id, json.dumps(data))
            minion = SparkMinion(redis_conn=redis_conn,
                    workflow_id=workflow_id, app_id=app_id,
                    config=config)
            minion.state = {
            }
            minion._process_message()

            # Discard first status message
            state_control.pop_app_output_queue(app_id, False)

            # First message is about missing state
            msg = json.loads(state_control.pop_app_output_queue(app_id, False))
            assert msg['status'] == 'WARNING', 'Invalid status'
            assert msg['code'] == SparkMinion.MNN003[0], 'Invalid code'
            
            # Second message is about invalid Python code
            msg = json.loads(state_control.pop_app_output_queue(app_id, False))
            assert msg['status'] == 'ERROR', 'Invalid status'
            assert msg.get('code') == SparkMinion.MNN006[0], 'Invalid code'

            # Third message is about unable to read data
            msg = json.loads(state_control.pop_app_output_queue(app_id, False))
            assert msg['status'] == 'ERROR', 'Invalid status'
            assert msg.get('code') == SparkMinion.MNN005[0], 'Invalid code'

            assert state_control.get_app_output_queue_size(
                app_id) == 0, 'There are messages in app output queue!'

            result = state_control.pop_queue(out_queue, False)
            assert result is None, 'Wrong CSV generated'


## def test_minion_perform_deliver_invalid_workflow_with_failure():
##    assert False, 'To be implemented'

# noinspection PyProtectedMember
def test_minion_perform_deliver_missing_state_invalid_port_failure():
    workflow_id = '6666'
    app_id = '5001'
    out_queue = 'queue_50001'
    task_id = 'f033f-284ab-28987e-232add'
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = get_side_effect(get_records(),
                                                           task_id, 0)
            redis_conn = mocked_redis()
            state_control = StateControlRedis(redis_conn)

            data = {
                'workflow_id': workflow_id,
                'app_id': app_id,
                'type': 'deliver',
                'task_id': task_id,
                'port': '2',  # This port is invalid
                'output': out_queue,
                'workflow': {"tasks": [], "flows":[]}
            }

            state_control.push_app_queue(app_id, json.dumps(data))
            minion = SparkMinion(redis_conn=redis_conn,
                    workflow_id=workflow_id, app_id=app_id,
                    config=config)
            minion.state = {}
            minion._process_message()
            
            # Discard first status message
            state_control.pop_app_output_queue(app_id, False)

            msg = json.loads(state_control.pop_app_output_queue(app_id, False))
            assert msg['status'] == 'WARNING', 'Invalid status'
            assert msg['code'] == SparkMinion.MNN003[0], 'Invalid code'

            msg = json.loads(state_control.pop_app_output_queue(app_id, False))
            assert msg['status'] == 'ERROR', 'Invalid status'
            assert msg.get('code') == SparkMinion.MNN004[0], 'Invalid code'

            result = state_control.pop_queue(out_queue, False)
            assert result is None, 'Wrong CSV generated'

# noinspection PyProtectedMember
def test_minion_perform_deliver_missing_state_unsupported_output_failure():
    workflow_id = '6666'
    app_id = '4001'
    out_queue = 'queue_40001'
    task_id = 'f033f-284ab-28987e-232add'
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = get_side_effect(get_records(),
                                                           task_id)
            redis_conn = mocked_redis()
            state_control = StateControlRedis(redis_conn)

            data = {
                'workflow_id': workflow_id,
                'app_id': app_id,
                'type': 'deliver',
                'task_id': task_id,
                'port': '1',
                'output': out_queue,
                'workflow': {"tasks": [], "flows":[]}
            }

            #state_control.push_app_delivery_queue(app_id, json.dumps(data))
            state_control.push_app_queue(app_id, json.dumps(data))
            minion = SparkMinion(redis_conn=redis_conn,
                    workflow_id=workflow_id, app_id=app_id,
                    config=config)
            minion.state = {
            }
            minion._process_message()
            
            # Discard first status message
            state_control.pop_app_output_queue(app_id, False)

            msg = json.loads(state_control.pop_app_output_queue(app_id, False))
            assert msg['status'] == 'WARNING', 'Invalid status'
            assert msg['code'] == SparkMinion.MNN003[0], 'Invalid code'

            msg = json.loads(state_control.pop_app_output_queue(app_id, False))
            assert msg['status'] == 'ERROR', 'Invalid status'
            assert msg.get('code') == SparkMinion.MNN001[0], 'Invalid code'

            result = state_control.pop_queue(out_queue, False)
            assert result is None, 'Wrong CSV generated'

            # check proper termination
            minion.terminate()
            assert not minion.is_spark_session_available()

def test_runner_minion_spark_configuration():
    """
    - Start a juicer server
    - Instanciate one minion passing specific configurations w.r.t. runtime
      environments (app_configs)
    - Assert that the spark context within the minion inherited the same configs
      that it was supposed to inherit
    """
    
    try:
        from pyspark.sql import SparkSession
    except ImportError as ie:
        # we will skip this test because pyspark is not installed
        return

    workflow_id = '6666'
    app_id = '897447'
    app_configs = {'spark.master': 'local[3]',
            'config1': '1', 'config2': '2'}
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = get_side_effect(None, 0, 1)

            redis_conn = mocked_redis()
            minion = SparkMinion(redis_conn=redis_conn,
                    workflow_id=workflow_id, app_id=app_id,
                    config=config)

            # Configure mocked redis
            state_control = StateControlRedis(redis_conn)
            with open(os.path.join(os.path.dirname(__file__),
                                   'fixtures/simple_workflow.json')) as f:
                data = json.loads(f.read())

            state_control.push_app_queue(app_id, json.dumps({
                'workflow_id': workflow_id,
                'app_id': app_id,
                'type': 'execute',
                'app_configs': app_configs,
                'workflow': data
                }))

            minion._process_message()

            # check spark session health
            assert not minion.spark_session is None
            assert minion.is_spark_session_available()

            # check configs
            ctx_configs = minion.spark_session.sparkContext.getConf().getAll()
            ctx_configs = {k:v for k,v in ctx_configs}
            for k,v in app_configs.iteritems():
                assert ctx_configs[k] == v

            # check app name
            assert minion.spark_session.sparkContext.appName == \
                    u'%s(workflow_id=%s,app_id=%s)' % (
                            data['name'], workflow_id, app_id)

            # check proper termination
            minion.terminate()
            assert not minion.is_spark_session_available()
