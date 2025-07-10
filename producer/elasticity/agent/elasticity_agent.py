from abc import ABC, abstractmethod

import pandas as pd

from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_manager import SloManager
from producer.request_handling.request_handler import RequestHandler
from producer.task_generation.task_generator import TaskGenerator


class ElasticityAgent(ABC):

    def __init__(self, elasticity_handler: ElasticityHandler, request_handler: RequestHandler, task_generator: TaskGenerator, track_slo_stats=True):
        self.elasticity_handler = elasticity_handler
        self.request_handler = request_handler
        self.task_generator = task_generator

        self._slo_manager = SloManager(elasticity_handler,
                                      request_handler,
                                      task_generator,
                                      queue_size_tolerance=2,
                                      avg_global_processing_t_tolerance=1,
                                      avg_worker_processing_t_tolerance=4,
                                      max_memory_usage= 0.9,
                                      track_stats=track_slo_stats)

        # possible observations and actions defined in Agent Implementation
    @abstractmethod
    def step(self):
        """
        Perform a single step of the agent

        Returns:
        """
        pass

    def get_slo_statistics(self) -> pd.DataFrame:
        return self._slo_manager.get_statistics()
