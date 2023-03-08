
# def set_enable_task(w, id_task, enable):
#     for i in range(len(w["workflow"]['tasks'])):
#         if w["workflow"]['tasks'][i]['id'] == id_task:
#             w["workflow"]['tasks'][i]['enabled'] = enable
#     return w


# def get_enabled_operation_id(w):
#     r = []
#     for i in range(len(w["workflow"]['tasks'])):
#         if w["workflow"]['tasks'][i]['enabled']:
#             r.append(w["workflow"]['tasks'][i]["operation"]['id'])
#     return r


# def get_task_by_comment(w, label):
#     tasks_list = []
#     for t in w['workflow']['tasks']:
#         if t['forms'].get('comment', {}).get('value', "") == label:
#             tasks_list.append([t['id'], t['operation']['id']])
#     return tasks_list
#
#
# def get_job_status(connection, job_id):
#     with connection.cursor() as cursor:
#         sql = """
#         SELECT status from stand.job WHERE id={}
#         """.format(job_id)
#         cursor.execute(sql)
#         result = cursor.fetchone()['status']
#     connection.commit()
#     return result


# def get_workflow(workflow_id):
#     url = 'http://localhost:23456/api/v1/tahiti/workflows/{}'.format(workflow_id)
#     headers = {"Content-Type": 'application/json;charset=UTF-8',
#                "X-Auth-Token": "123456"}
#     r = requests.get(url, headers=headers)
#     return {'workflow': json.loads(r.text)}
#
#
# def get_operation(op_id):
#     url = 'http://localhost:23456/api/v1/tahiti/operations/{}'.format(op_id)
#     headers = {"Content-Type": 'application/json;charset=UTF-8',
#                "X-Auth-Token": "123456"}
#     r = requests.get(url, headers=headers)
#     return json.loads(r.text)


# def remove_flow(w, task):
#     task_id = task[0]
#
#     flow = []
#     for f in w['workflow']['flows']:
#
#         if task_id not in [f['source_id'], f['target_id']]:
#             flow.append(f)
#
#     w['workflow']['flows'] = flow
#
#     return w
#
#
# def set_flow(w, task, save):
#     task_id = task[1]
#     ports = get_operation(task_id)['ports']
#     port = -1
#     name = 'output data'
#
#     for p in ports:
#         if p['name'] == name:
#             port = p['id']
#             break
#
#     f = {'source_port': port,
#          'target_port': save[2],
#          'source_port_name': 'input data',
#          'target_port_name': 'output data',
#          'environment': 'DESIGN',
#          'source_id': task[0],
#          'target_id': save[0]
#          }
#
#     w['workflow']['flows'].append(f)
#     w = set_enable_task(w, save[0], True)
#
#     return w


# def run_workflow(data, workflow_id):
#     url = 'http://localhost:23456/api/v1/stand/jobs'
#     headers = {"Content-Type": 'application/json;charset=UTF-8',
#                "Referer": "http://localhost:23456/workflows/1/{}".format(workflow_id),
#                "X-Auth-Token": "123456"}
#
#     data['cluster'] = {'id': 1}
#     r = requests.post(url, data=json.dumps(data), headers=headers)
#     r = json.loads(r.text)
#     status = r['status']
#     job_id = -1
#     if status != "ERROR":
#         job_id = r['data']['id']
#     else:
#         print(r)
#     return job_id, status


# def get_valid_job(job_id):
#     with connection.cursor() as cursor:
#         sql = """
#         SELECT * FROM {STAND_DB}.job where id = {JOB_ID};
#         """.format(STAND_DB=STAND_DB, JOB_ID=job_id)
#         cursor.execute(sql)
#         result = cursor.fetchone()
#     connection.commit()
#     if result["status"] == "COMPLETED":
#         return result
#     else:
#         print("Job {} can not be considered (status: {}).".format(job_id, result["status"]))