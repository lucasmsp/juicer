from textwrap import dedent

from juicer.spark.vis_operation import HtmlVisualizationModel


class BaseHtmlReport(object):
    pass


class EvaluateModelOperationReport(BaseHtmlReport):
    """ Report generated by class EvaluateModelOperation """

    # noinspection PyProtectedMember
    @staticmethod
    def generate_visualization(**kwargs):
        evaluator = kwargs['evaluator']
        title = kwargs['title']
        operation_id = kwargs['operation_id']
        task_id = kwargs['task_id']
        metric_value = kwargs['metric_value']
        metric_name = kwargs['metric_name']

        all_params = [
            ('<tr>'
             '  <td>{name}</td><td><em>{doc}</em></td>'
             '  <td>{value}</td><td>{default}</td>'
             '</tr>').format(name=x.name, doc=x.doc,
                             value=evaluator._paramMap.get(x, 'unset'),
                             default=evaluator._defaultParamMap.get(
                                 x, 'unset')) for x in
            evaluator.extractParamMap()]

        vis_model = dedent('''
            <p>
               <h5>Evaluation result</h5>
               <h6>{metric_name}: {metric_value:.5g}</h6>
            </p>
            <table class="table table-bordered table-striped table-sm"
                style="width:100%">
              <thead>
                 <tr>
                   <th>Parameter</th>
                   <th>Description</th>
                   <th>Value</th>
                   <th>Default</th>
                 </tr>
              </thead>
              <tbody>
                {params}
              </tbody>
            </table>
        ''').format(title=title, params=''.join(all_params),
                    metric_value=metric_value,
                    metric_name=metric_name)

        return HtmlVisualizationModel(
            vis_model, task_id, operation_id, 'EvaluateModelOperation', title,
            '[]', '', '', '', {})
