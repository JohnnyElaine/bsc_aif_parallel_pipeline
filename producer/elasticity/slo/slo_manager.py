import pandas as pd

from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_status import SloStatus
from producer.elasticity.slo.slo_util import SloUtil
from producer.elasticity.slo.values.avg_global_processing_time_slo import AvgGlobalProcessingTimeSlo
from producer.elasticity.slo.values.avg_worker_processing_time_slo import AvgWorkerProcessingTimeSlo
from producer.elasticity.slo.values.memory_slo import MemorySlo
from producer.elasticity.slo.values.queue_size_slo import QueueSizeSlo
from producer.request_handling.request_handler import RequestHandler
from producer.statistics.slo_statistics import SloStatistics
from producer.task_generation.task_generator import TaskGenerator


class SloManager:

    def __init__(self,
                 elasticity_handler: ElasticityHandler,
                 request_handler: RequestHandler,
                 task_generator: TaskGenerator,
                 queue_size_tolerance=2,
                 avg_global_processing_t_tolerance=1,
                 avg_worker_processing_t_tolerance=4,
                 max_memory_usage=0.9,
                 track_stats=True):
        self._elasticity_handler = elasticity_handler
        self._task_generator = task_generator

        self._stats = None
        if track_stats:
            self._stats = SloStatistics()

        self._queue_size_slo = QueueSizeSlo(task_generator, tolerance=queue_size_tolerance, stats=self._stats)
        self._memory_slo = MemorySlo(max_memory_usage, stats=self._stats)
        # TODO: extract stats (MovingAverage) from request_handler so we dont have to pass entire reference to request_handler
        self._avg_global_processing_time_slo = AvgGlobalProcessingTimeSlo(request_handler, task_generator, tolerance=avg_global_processing_t_tolerance, stats=self._stats)
        self._avg_worker_processing_time_slo = AvgWorkerProcessingTimeSlo(request_handler, task_generator, tolerance=avg_worker_processing_t_tolerance, stats=self._stats)

    def get_all_slo_probabilities(self) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Get probabilities for all 4 SLOs.
        
        Returns:
            tuple: (queue_probs, memory_probs, global_processing_probs, worker_processing_probs)
                   Each element is a list [P(OK), P(WARNING), P(CRITICAL)]
        """
        return (
            self._queue_size_slo.probabilities(),
            self._memory_slo.probabilities(),
            self._avg_global_processing_time_slo.probabilities(),
            self._avg_worker_processing_time_slo.probabilities()
        )

    def get_all_slo_status(self) -> tuple[SloStatus, SloStatus, SloStatus, SloStatus]:
        qsize, mem, global_processing_t, worker_processing_t = self.get_all_slo_values()

        return SloUtil.get_slo_status(qsize), SloUtil.get_slo_status(mem), SloUtil.get_slo_status(global_processing_t), SloUtil.get_slo_status(worker_processing_t)
    
    def get_all_slo_values(self) -> tuple[float, float, float, float]:
        qsize_slo_value = self._queue_size_slo.value()
        mem_slo_value = self._memory_slo.value()
        global_avg_processing_time_value = self._avg_global_processing_time_slo.value()
        worker_avg_processing_time_value = self._avg_worker_processing_time_slo.value()

        if self._stats is not None:
            self._track_capacity_stats()

        return qsize_slo_value, mem_slo_value, global_avg_processing_time_value, worker_avg_processing_time_value

    def get_statistics(self) -> pd.DataFrame:
        return self._stats.to_dataframe()

    def _track_capacity_stats(self):
        self._stats.fps_capacity.append(self._elasticity_handler.state_fps.capacity())
        self._stats.resolution_capacity.append(self._elasticity_handler.state_resolution.capacity())
        self._stats.inference_quality.append(self._elasticity_handler.state_inference_quality.capacity())
