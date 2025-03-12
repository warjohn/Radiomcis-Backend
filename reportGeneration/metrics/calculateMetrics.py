import sklearn as sk
import asyncio

class CalculateMetrics():

    def __init__(self, metrics, flag):
        self.metrics = metrics
        self.flag = flag


    def calculate(self, y_trueLabel, predictions):
        metrics_result = {}
        tmp_task = []
        loop = asyncio.get_event_loop()
        for metric in self.metrics:
            task = loop.create_task(self.__caluclate(metric, metrics_result, y_trueLabel, predictions))
            tmp_task.append(task)
        loop.run_until_complete(asyncio.gather(*tmp_task))
        return metrics_result

    async def __caluclate(self, metric, metrics, y_trueLabel, predictions):
        metric_func = getattr(sk.metrics, metric['name'])
        params = metric['params']
        metrics[metric['name']] = metric_func(y_trueLabel, predictions, **params)
