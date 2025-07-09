from producer.elasticity.slo.slo_util import SloUtil
from producer.request_handling.request_handler import RequestHandler
from producer.task_generation.task_generator import TaskGenerator

class AvgWorkerProcessingTimeSlo:

    def __init__(self, request_handler: RequestHandler, task_generator: TaskGenerator, tolerance=1, stats=None):
        self._request_handler = request_handler
        self._task_generator = task_generator
        self._tolerance = tolerance
        self._stats = stats

    def value(self, track_stats=True):
        avg_processing_time = self._request_handler.avg_global_processing_time()
        value = avg_processing_time / self._task_generator.frame_time() * self._tolerance

        if track_stats and (self._stats is not None):
            self._stats.avg_global_processing_time.append(avg_processing_time)
            self._stats.avg_global_processing_time_slo_value.append(value)

        return value

    def probabilities(self) -> list:
        """
        Calculate probabilities for each queue SLO state (OK, WARNING, CRITICAL).
        Uses the value of current queue size to maximum allowed queue size.

        Returns:
            list: Probabilities for each SLO state [p_ok, p_warning, p_critical]
        """
        return SloUtil.get_slo_state_probabilities(self.value(track_stats=False))