from abc import ABC, abstractmethod

from producer.elasticity.agent.action.action_type import ActionType
from producer.elasticity.handler.elasticity_handler import ElasticityHandler


class ElasticityAgent(ABC):

    def __init__(self, elasticity_handler: ElasticityHandler):
        self.elasticity_handler = elasticity_handler

    @abstractmethod
    def step(self) -> tuple[ActionType, bool]:
        """
        Perform a single step of the agent

        Returns:
            tuple[ActionType, bool]: The action taken and whether it was successful
        """
        pass
