from producer.elasticity.agent.aif.aif_agent import ActiveInferenceAgent
from producer.elasticity.agent.aif.aif_agent_experimental_1 import ActiveInferenceAgentExperimental1
from producer.elasticity.agent.aif.aif_agent_experimental_2 import ActiveInferenceAgentExperimental2
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.agent.heuristic.heuristic_agent import HeuristicAgent
from producer.elasticity.agent.rl.rl_agent import ReinforcementLearningAgent
from producer.elasticity.agent.test.test_agent import TestAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.enums.agent_type import AgentType
from producer.task_generation.task_generator import TaskGenerator


class AgentFactory:
    @staticmethod
    def create(agent_type: AgentType, elasticity_handler: ElasticityHandler, task_generator: TaskGenerator) -> ElasticityAgent | None:
        match agent_type:
            case AgentType.NONE:
                return None
            case AgentType.TEST:
                return TestAgent(elasticity_handler, task_generator)
            case AgentType.ACTIVE_INFERENCE:
                return ActiveInferenceAgent(elasticity_handler, task_generator)
            case AgentType.HEURISTIC:
                return HeuristicAgent(elasticity_handler, task_generator)
            case AgentType.REINFORCEMENT_LEARNING:
                return ReinforcementLearningAgent(elasticity_handler, task_generator)
            case AgentType.ACTIVE_INFERENCE_EXPERIMENTAL_1:
                return ActiveInferenceAgentExperimental1(elasticity_handler, task_generator)
            case AgentType.ACTIVE_INFERENCE_EXPERIMENTAL_2:
                return ActiveInferenceAgentExperimental2(elasticity_handler, task_generator)
            case _:
                return ActiveInferenceAgent(elasticity_handler, task_generator)