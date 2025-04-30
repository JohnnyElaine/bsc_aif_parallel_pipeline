from abc import ABC, abstractmethod

import pandas as pd

from producer.elasticity.action.general_action_type import GeneralActionType
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_manager import SloManager


class ElasticityAgent(ABC):

    def __init__(self, elasticity_handler: ElasticityHandler):
        self.elasticity_handler = elasticity_handler
        self.slo_manager = SloManager(self.elasticity_handler, max_queue_size=self.elasticity_handler.max_fps * 3,
                                      max_memory_usage= 0.9)

    @abstractmethod
    def step(self):
        """
        Perform a single step of the agent

        Returns:
        """
        pass

    def get_slo_statistics(self) -> pd.DataFrame:
        return self.slo_manager.get_statistics()
