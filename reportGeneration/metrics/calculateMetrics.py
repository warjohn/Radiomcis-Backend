import sklearn as sk

class CalculateMetrics():

    def __init__(self, metrics, flag):
        self.metrics = metrics
        self.flag = flag

    def calculate(self, y_trueLabel, predictions):
        metrics_result = {}
        result = None
        for metric in self.metrics:
            result = self.__caluclate(metric, metrics_result, y_trueLabel, predictions)
        return result

    def __caluclate(self, metric, metrics, y_trueLabel, predictions):
        metric_func = getattr(sk.metrics, metric['name'])
        params = metric['params']
        metrics[metric['name']] = metric_func(y_trueLabel, predictions, **params)
        return metrics
