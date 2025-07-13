from producer.elasticity.slo.slo_util import SloUtil
from producer.request_handling.request_handler import RequestHandler
from producer.task_generation.task_generator import TaskGenerator

class AvgWorkerProcessingTimeSlo:
    # TODO add documentation describing the purpose
    def __init__(self, request_handler: RequestHandler, task_generator: TaskGenerator, tolerance=1, stats=None):
        self._request_handler = request_handler
        self._task_generator = task_generator
        self._tolerance = tolerance
        self._stats = stats

    def value(self, track_stats=True):
        # key=worker-addr, value=processing-time
        avg_worker_processing_times_dict = self._request_handler.avg_worker_processing_times()

        if not avg_worker_processing_times_dict:
            return 0

        highest_avg_processing_t = max(avg_worker_processing_times_dict.values())

        value = highest_avg_processing_t / (self._task_generator.frame_time * self._tolerance)

        if track_stats and (self._stats is not None):
            self._stats.avg_worker_processing_time_slo_value.append(value)

            for addr, avg_processing_t in avg_worker_processing_times_dict.items():
                # Consider finding better place for single initiation, so we don't have to check every time a new value is added
                if not addr in self._stats.avg_worker_processing_time:
                    self._stats.avg_worker_processing_time[addr] = []
                self._stats.avg_worker_processing_time[addr].append(avg_processing_t)


        return value

    def probabilities(self) -> list:
        """
        Calculate probabilities for each queue SLO state (OK, WARNING, CRITICAL).
        Uses the value of current queue size to maximum allowed queue size.

        Returns:
            list: Probabilities for each SLO state [p_ok, p_warning, p_critical]
        """
        return SloUtil.get_slo_state_probabilities(self.value(track_stats=False))