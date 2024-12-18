
import json
import os
import uuid
from textwrap import dedent, indent

import datetime

from juicer.operation import Operation
from juicer.service import limonero_service

from urllib.request import urlopen
from urllib.parse import urlparse, parse_qs

from juicer.scikit_learn.data_operation import DataReaderOperation


class DataReaderOperationCUDF(DataReaderOperation):

    def __init__(self, parameters, named_inputs, named_outputs):
        DataReaderOperation.__init__(self, parameters,  named_inputs,  named_outputs)

    def generate_code(self):
        """
        """
        if not self.has_code:
            return ''

        infer_from_data = self.infer_schema == self.INFER_FROM_DATA
        infer_from_limonero = self.infer_schema == self.INFER_FROM_LIMONERO
        do_not_infer = self.infer_schema == self.DO_NOT_INFER
        mode_failfast = self.mode == self.OPT_MODE_FAILFAST

        protect = (self.parameters.get('export_notebook', False) or
                   self.parameters.get('plain', False)) or self.plain
        data_format = self.metadata.get('format')

        parsed = urlparse(self.metadata['url'])

        extra_params = {}
        if 'extra_params' in self.metadata['storage']:
            if self.metadata['storage']['extra_params']:
                extra_params = json.loads(self.metadata['storage'][
                                              'extra_params'])

        if self.metadata.get('privacy_aware', False):
            raise ValueError(_('Not supported'))

        if parsed.scheme not in ('hdfs', 'file', 'mysql'):
            raise ValueError(_('Not supported'))

        if data_format not in ('CSV', 'TEXT', 'PARQUET', 'JDBC', 'JSON'):
            raise ValueError(_('Not supported'))

        if data_format == 'JDBC':
            qs_parsed = parse_qs(parsed.query)
            if parsed.scheme not in self.SUPPORTED_DRIVERS:
                raise ValueError(
                    _('Database {} not supported').format(parsed.scheme))
            if not self.metadata.get('command'):
                raise ValueError(
                    _('No command nor table specified for data source.'))

            jdbc_code = indent(dedent(self.SUPPORTED_DRIVERS[parsed.scheme].format(
                scheme=parsed.scheme, host=parsed.hostname,
                db=parsed.path[1:],
                port=parsed.port,
                query=self.metadata.get('command'),
                user=qs_parsed.get('user', [''])[0],
                password=qs_parsed.get('password', [''])[0],
                out=self.output)), '    ')
        else:
            jdbc_code = None

        self.header = self.metadata.get('is_first_line_header')

        attributes, converters, parse_dates, names = self.analyse_attributes(
            self.metadata.get('attributes'))

        self.template = """
            {%- if infer_from_limonero %}
            {%-   if attributes and format in ('TEXT', 'CSV') %}
            columns = {
            {%-     for attr in attributes %}
                '{{attr[0]}}': {{attr[1]}},
            {%-     endfor %}
            }
            {%-   elif format in ('TEXT', 'CSV') %}
            columns = {'value': object}
            {%-   endif %}
            {%- elif infer_from_data and format in ('TEXT', 'CSV') %}
            columns = None
            header = 'infer'
            {%- elif do_not_infer and format in ('TEXT', 'CSV') %}
            header = 'infer'
            {%- endif %}

            # Open data source
            {%- if protect %}
            f = open('{{parsed.path.split('/')[-1]}}', 'rb')
            {%- elif parsed.scheme == 'hdfs'  %}
            fs = hdfs.HadoopFileSystem('{{parsed.hostname}}', {{parsed.port}}, 
               user='{{extra_params.get('user', parsed.username) or 'hadoop'}}')
            f = fs.open_input_file('{{parsed.path}}')
            {%- elif parsed.scheme == 'file' %}
            f = open('{{parsed.path}}', 'rb')
            {%- endif %}

            {%- if format == 'CSV' %}
            {{output}} = cudf.read_csv(f, sep='{{sep}}',
                                     header={{header}},
                                     {%- if infer_from_limonero %}
                                     names={{names}},
                                     dtype=columns,
                                     parse_dates={{parse_dates}},
                                     #converters={{converters}},
                                     {%- elif do_not_infer %}
                                     parse_dates = None,
                                     dtype='str',
                                     {%-   endif %}
                                     na_values={{na_values}})
            f.close()
            {%-   if header == 'infer' %}
            {{output}}.columns = ['attr{{i}}'.format(i=i) 
                            for i, _ in enumerate({output}.columns)]
            {%-   endif %}
            {%- elif format == 'TEXT' %}
            {{output}} = pd.read_csv(f, sep='{{sep}}',
                                     encoding='{{encoding}}',
                                     names = ['value'],
                                     error_bad_lines={{mode_failfast}})
            f.close()
            {%- elif format == 'PARQUET' %}
            {{output}} = pd.read_parquet(f, engine='pyarrow')
            f.close()
            {%- elif format == 'JSON' %}
            {{output}} = pd.read_json(f, lines=True)
            f.close()
            {%- elif format == 'JDBC' %}
            {{jdbc_code}}
            {%- endif %}

            {%- if infer_from_data %}
            {{output}} = {{output}}.infer_objects()
            {%- endif %}
            n_rows = "Records: " + str(len({{output}}))
            emit_event(name='update task', message=_(n_rows), identifier=task_id, status='RUNNING')

            """
        ctx = {
            'attributes': attributes,
            'parse_dates': parse_dates,
            'names': names,
            'converters': converters,

            'infer_from_limonero': infer_from_limonero,
            'infer_from_data': infer_from_data,
            'do_not_infer': do_not_infer,
            'is_first_line_header': self.header,

            'protect': protect,  # Hide information about path
            'parsed': parsed,
            'extra_params': extra_params,
            'format': data_format,
            'encoding': self.metadata.get('encoding', 'utf-8') or 'utf-8',
            'header': 0 if self.header else 'None',
            'sep': self.sep,
            'na_values': self.null_values if len(self.null_values) else 'None',
            'output': self.output,
            'mode_failfast': mode_failfast,

            'jdbc_code': jdbc_code
        }
        return dedent(self.render_template(ctx))