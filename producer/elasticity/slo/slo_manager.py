import pandas as pd
import psutil

from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_status import SloStatus
from producer.statistics.slo_statistics import SloStatistics


class SloManager:

    CRITICAL_THRESHOLD = 1 # below this threshold the SLO is satisfied
    WARNING_THRESHOLD = 0.8

    def __init__(self,
                 elasticity_handler: ElasticityHandler,
                 max_queue_size=60,
                 max_memory_usage=0.8):
        self._elasticity_handler = elasticity_handler
        self._statistics = SloStatistics()

        # Critical thresholds (SLO unsatisfied)
        self._max_qsize = max_queue_size
        self._max_mem_usage = max_memory_usage

    def get_statistics(self) -> pd.DataFrame:
        return self._statistics.to_dataframe()

    def get_all_slo_status(self, track_statistics=False):
        qsize_slo_status = self.get_qsize_slo_status(track_statistics=track_statistics)
        mom_slo_status = self.get_mem_slo_status(track_statistics=track_statistics)

        if track_statistics:
            self._statistics.fps_capacity.append(self._elasticity_handler.state_fps.get_capacity())
            self._statistics.resolution_capacity.append(self._elasticity_handler.state_resolution.get_capacity())
            self._statistics.work_load_capacity.append(self._elasticity_handler.state_work_load.get_capacity())


        return qsize_slo_status, mom_slo_status

    def get_qsize_slo_status(self, track_statistics=False) -> SloStatus:
        """
        Get the current queue SLO status (SATISFIED, WARNING, or UNSATISFIED).

        Returns:
            SloStatus: The current status of the queue SLO
        """
        return SloManager.get_slo_status(self.get_qsize_ratio(track_statistics=track_statistics))

    def get_mem_slo_status(self, track_statistics=False) -> SloStatus:
        """
        Get the current memory SLO status (SATISFIED, WARNING, or UNSATISFIED).

        Returns:
            SloStatus: The current status of the memory SLO
        """
        return SloManager.get_slo_status(self.get_mem_ratio(track_statistics=track_statistics))
    
    def get_qsize_ratio(self, track_statistics=False):
        queue_size = self._elasticity_handler.queue_size()
        ratio = queue_size / self._max_qsize

        if track_statistics:
            self._statistics.queue_size.append(queue_size)
            self._statistics.queue_size_slo_ratio.append(ratio)

        return ratio

    def get_mem_ratio(self, track_statistics=False):
        mem_usage = psutil.virtual_memory().percent / 100
        ratio = mem_usage / self._max_mem_usage

        if track_statistics:
            self._statistics.memory_usage.append(mem_usage)
            self._statistics.memory_usage_slo_ratio.append(ratio)

        return ratio

    def get_queue_slo_state_probabilities(self) -> list:
        """
        Calculate probabilities for each queue SLO state (SATISFIED, WARNING, UNSATISFIED).
        Uses the ratio of current queue size to maximum allowed queue size.

        Returns:
            list: Probabilities for each SLO state [p_satisfied, p_warning, p_unsatisfied]
        """
        return SloManager.get_slo_state_probabilities(self.get_qsize_ratio())

    def get_memory_slo_state_probabilities(self) -> list:
        """
        Calculate probabilities for each memory SLO state (SATISFIED, WARNING, UNSATISFIED).
        Uses the ratio of current memory usage to maximum allowed memory usage.

        Returns:
            list: Probabilities for each SLO state [p_satisfied, p_warning, p_unsatisfied]
        """
        return SloManager.get_slo_state_probabilities(self.get_mem_ratio())

    @staticmethod
    def get_slo_state_probabilities(ratio: float):
        if ratio <= SloManager.WARNING_THRESHOLD:
            # In satisfied region (OK zone)
            # As ratio approaches WARNING_THRESHOLD, satisfaction probability decreases
            p_ok = 1.0 - (ratio / SloManager.WARNING_THRESHOLD) * 0.2  # 0.8-1.0 range
            p_warning = 1.0 - p_ok
            p_critical = 0.0
        elif ratio <= SloManager.CRITICAL_THRESHOLD:
            # In warning region
            # As ratio approaches CRITICAL_THRESHOLD, warning probability decreases
            range_size = SloManager.CRITICAL_THRESHOLD - SloManager.WARNING_THRESHOLD
            position = ratio - SloManager.WARNING_THRESHOLD
            p_warning = 1.0 - (position / range_size) * 0.2  # 0.8-1.0 range
            p_critical = 1.0 - p_warning
            p_ok = 0.0
        else:
            # In unsatisfied region (CRITICAL zone)
            p_critical = 1.0
            p_ok = 0.0
            p_warning = 0.0
        return [p_ok, p_warning, p_critical]


    @staticmethod
    def get_slo_status(ratio: float):
        if ratio <= SloManager.WARNING_THRESHOLD: # Safe zone
            return SloStatus.OK
        elif ratio <= SloManager.CRITICAL_THRESHOLD: # warning zone
            return SloStatus.WARNING
        else:
            return SloStatus.CRITICAL # critical zone (slo not satisfied)

