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
        self.slo_manager = SloManager(self.elasticity_handler,
                                      task_generator,
                                      queue_size_tolerance=self.elasticity_handler.max_fps * 3,
                                      max_memory_usage= 0.9,
                                      track_stats=track_slo_stats)

    @abstractmethod
    def step(self):
        """
        Perform a single step of the agent

        Returns:
        """
        pass

    def get_slo_statistics(self) -> pd.DataFrame:
        return self.slo_manager.get_statistics()
