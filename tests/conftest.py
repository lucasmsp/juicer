# from project.database import db as _db
import logging
import sys
from juicer.service import limonero_service

import os
import pytest
import gettext

sys.path.append(os.path.dirname(os.path.curdir))

def pytest_sessionstart(session):
    locales_path = os.path.join(os.path.dirname(__file__), '..', 'juicer', 
        'i18n', 'locales')
    t = gettext.translation('messages', locales_path, ['en'],
                            fallback=True)
    t.install()

# Mock for Limonero services
def patched_get_data_source_info(base_url, token, data_source_id):
    return {
        'attributes': [],
        'format': 'CSV',
        'url': 'http://hdfs.lemonade:9000'
    }


limonero_service.get_data_source_info = patched_get_data_source_info


@pytest.fixture(scope='function')
def app(request):
    """Session-wide test `Flask` application."""
    settings_override = {
        'TESTING': True,
    }
    yield settings_override
