from producer.elasticity.agent.action.action_type import ActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler


class HeuristicAgent(ElasticityAgent):

    def __init__(self, elasticity_handler: ElasticityHandler):
        super().__init__(elasticity_handler)

    def step(self) -> tuple[ActionType, bool]:
        pass