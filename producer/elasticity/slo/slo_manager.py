import pandas as pd
import psutil
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.statistics.slo_statistics import SloStatistics


class SloManager:
    def __init__(self, elasticity_handler: ElasticityHandler,
                 max_memory_percentage = 80):
        self._elasticity_handler = elasticity_handler
        self._slo_statistics = SloStatistics()
        self._max_memory_percentage = max_memory_percentage

    def is_queue_slo_satisfied(self) -> bool:
        """
            Check if the current queue size satisfies the Service Level Objective (SLO).

            Returns:
                True if the queue size is within the allowed limit, False otherwise.
        """

        is_satisfied, queue_size = self._check_queue_slo()

        self._slo_statistics.queue_slo_satisfaction.append(is_satisfied)
        self._slo_statistics.queue_sizes.append(queue_size)

        return is_satisfied

    def is_memory_slo_satisfied(self) -> bool:
        """
            Check if the current memory usage satisfies the Service Level Objective (SLO).

            Returns:
                True if memory usage is within the allowed threshold, False otherwise.
        """
        is_satisfied, memory_usage_percent = self._check_queue_slo()

        self._slo_statistics.memory_slo_satisfaction.append(is_satisfied)
        self._slo_statistics.memory_usage.append(memory_usage_percent)

        return is_satisfied

    def probability_queue_slo_satisfied(self) -> float:
        return 1 - self._elasticity_handler.queue_size() / self._max_qsize()

    def probability_memory_slo_satisfied(self) -> float:
        return 1 - psutil.virtual_memory().percent / self._max_memory_percentage

    def _max_qsize(self):
        return 2 * self._elasticity_handler.fps # Example: 2 seconds worth of frames

    def _check_queue_slo(self) -> tuple[bool, int]:
        """
            Check if the current queue size satisfies the Service Level Objective (SLO).

            Returns:
                tuple[bool, int]: A tuple where:
                    - First element (bool): True if the queue size is within the allowed limit, False otherwise.
                    - Second element (int): The current queue size.
            """

        queue_size = self._elasticity_handler.queue_size()
        return queue_size <= self._max_qsize(), queue_size

    def _check_memory_slo(self) -> tuple[bool, float]:
        """
            Check if the current memory usage satisfies the Service Level Objective (SLO).

            Returns:
                tuple[bool, float]: A tuple where:
                    - First element (bool): True if memory usage is within the allowed threshold, False otherwise.
                    - Second element (float): Current memory usage percentage (0-100).
            """
        memory_usage_percent = psutil.virtual_memory().percent
        return memory_usage_percent <= self._max_memory_percentage, memory_usage_percent

    def get_statistics(self) -> pd.DataFrame:
        return self._slo_statistics.to_dataframe()