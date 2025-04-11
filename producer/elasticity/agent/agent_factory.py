from producer.elasticity.agent.aif.aif_agent import ActiveInferenceAgent
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.agent.heuristic.heuristic_agent import HeuristicAgent
from producer.elasticity.agent.rl.rl_agent import ReinforcementLearningAgent
from producer.elasticity.agent.test.test_agent import TestAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.enums.agent_type import AgentType


class AgentFactory:
    @staticmethod
    def create(agent_type: AgentType, elasticity_handler: ElasticityHandler) -> ElasticityAgent | None:
        match agent_type:
            case AgentType.NONE:
                return None
            case AgentType.TEST:
                return TestAgent(elasticity_handler)
            case AgentType.ACTIVE_INFERENCE:
                return ActiveInferenceAgent(elasticity_handler)
            case AgentType.HEURISTIC:
                return HeuristicAgent(elasticity_handler)
            case AgentType.REINFORCEMENT_LEARNING:
                return ReinforcementLearningAgent(elasticity_handler)
            case _:
                return ActiveInferenceAgent(elasticity_handler)