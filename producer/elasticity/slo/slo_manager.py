import pandas as pd
import psutil

from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_status import SloStatus
from producer.statistics.slo_statistics import SloStatistics
from producer.task_generation.task_generator import TaskGenerator


# TODO: Split SLOs into their own file
# TODO: Add 2 new SLOs, avg processing time, avg processing time per worker
class SloManager:

    CRITICAL_THRESHOLD = 1 # below this threshold the SLO is satisfied
    WARNING_THRESHOLD = 0.9

    def __init__(self,
                 elasticity_handler: ElasticityHandler,
                 task_generator: TaskGenerator,
                 max_queue_size=60,
                 max_memory_usage=0.8):
        self._elasticity_handler = elasticity_handler
        self._task_generator = task_generator
        self._statistics = SloStatistics()

        # Critical thresholds (SLO unsatisfied)
        self._max_qsize = max_queue_size
        self._max_mem_usage = max_memory_usage

    def get_statistics(self) -> pd.DataFrame:
        return self._statistics.to_dataframe()

    def get_all_slo_status(self, track_stats=False) -> tuple[SloStatus, SloStatus]:
        qsize_slo_value, mem_slo_value = self.get_all_slo_values(track_stats=track_stats)

        return SloManager.get_slo_status(qsize_slo_value), SloManager.get_slo_status(mem_slo_value)
    
    def get_all_slo_values(self, track_stats=False) -> tuple[float, float]:
        qsize_slo_value = self.get_qsize_value(track_stats=track_stats)
        mem_slo_value = self.get_mem_value(track_stats=track_stats)
        
        if track_stats:
            self._track_capacity_stats()

        return qsize_slo_value, mem_slo_value
    
    def get_qsize_value(self, track_stats=False):
        queue_size = self._task_generator.queue_size()
        value = queue_size / self._max_qsize

        if track_stats:
            self._statistics.queue_size.append(queue_size)
            self._statistics.queue_size_slo_value.append(value)

        return value

    def get_mem_value(self, track_stats=False):
        mem_usage = psutil.virtual_memory().percent / 100
        value = mem_usage / self._max_mem_usage

        if track_stats:
            self._statistics.memory_usage.append(mem_usage)
            self._statistics.memory_usage_slo_value.append(value)

        return value

    def get_qsize_slo_state_probabilities(self) -> list:
        """
        Calculate probabilities for each queue SLO state (OK, WARNING, CRITICAL).
        Uses the value of current queue size to maximum allowed queue size.

        Returns:
            list: Probabilities for each SLO state [p_ok, p_warning, p_critical]
        """
        return SloManager._get_slo_state_probabilities(self.get_qsize_value())

    def get_mem_slo_state_probabilities(self) -> list:
        """
        Calculate probabilities for each memory SLO state (OK, WARNING, CRITICAL).
        Uses the value of current memory usage to maximum allowed memory usage.

        Returns:
            list: Probabilities for each SLO state [p_ok, p_warning, p_critical]
        """
        return SloManager._get_slo_state_probabilities(self.get_mem_value())
    
    def _track_capacity_stats(self):
        self._statistics.fps_capacity.append(self._elasticity_handler.state_fps.capacity())
        self._statistics.resolution_capacity.append(self._elasticity_handler.state_resolution.capacity())
        self._statistics.inference_quality.append(self._elasticity_handler.state_inference_quality.capacity())
    
    @staticmethod
    def get_slo_status(value: float):
        if value <= SloManager.WARNING_THRESHOLD: # Safe zone
            return SloStatus.OK
        elif value <= SloManager.CRITICAL_THRESHOLD: # warning zone
            return SloStatus.WARNING
        else:
            return SloStatus.CRITICAL # critical zone (slo not satisfied)

    @staticmethod
    def _get_slo_state_probabilities(value: float):
        if value <= SloManager.WARNING_THRESHOLD:
            # In satisfied region (OK zone)
            # As value approaches WARNING_THRESHOLD, satisfaction probability decreases
            p_ok = 1.0 - (value / SloManager.WARNING_THRESHOLD) * 0.2  # 0.8-1.0 range
            p_warning = 1.0 - p_ok
            p_critical = 0.0
        elif value <= SloManager.CRITICAL_THRESHOLD:
            # In warning region
            # As value approaches CRITICAL_THRESHOLD, warning probability decreases
            range_size = SloManager.CRITICAL_THRESHOLD - SloManager.WARNING_THRESHOLD
            position = value - SloManager.WARNING_THRESHOLD
            p_warning = 1.0 - (position / range_size) * 0.2  # 0.8-1.0 range
            p_critical = 1.0 - p_warning
            p_ok = 0.0
        else:
            # In unsatisfied region (CRITICAL zone)
            p_critical = 1.0
            p_ok = 0.0
            p_warning = 0.0
        return [p_ok, p_warning, p_critical]