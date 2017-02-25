# -*- coding: utf-8 -*-
import json
import logging
import requests
import urlparse

log = logging.getLogger()
log.setLevel(logging.DEBUG)

def query_caipirinha(base_url, item_path, token, item_id, data):
    headers = {'X-Auth-Token': token}

    if item_id == '':
        url = '{}/{}'.format(base_url, item_path)
    else:
        url = '{}/{}/{}'.format(base_url, item_path, item_id)

    log.debug('Querying Caipirinha URL: %s', url)

    r = requests.post(url, headers=headers, data=data)
    if r.status_code == 200:
        return json.loads(r.text)
    else:
        raise RuntimeError(
            u"Error loading storage id {}: HTTP {} - {}".format(item_id,
                                                                r.status_code,
                                                                r.text))

def new_dashboard(base_url, token, title, user,
        workflow_id, workflow_name, job_id, task_id, visualizations):
    data = dict(title=title, user=user, workflow_id=workflow_id,
            workflow_name=workflow_name, job_id=job_id, task_id=task_id,
            visualizations=visualizations)
    return query_caipirinha(base_url, 'dashboards', token, '', json.dumps(data))
