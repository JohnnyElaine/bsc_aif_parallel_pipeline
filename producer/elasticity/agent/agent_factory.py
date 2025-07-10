from producer.elasticity.agent.aif.aif_agent_relative_control import ActiveInferenceAgentRelativeControl
from producer.elasticity.agent.aif.aif_agent_experimental_1 import ActiveInferenceAgentExperimental1
from producer.elasticity.agent.aif.aif_agent_absolute_control import ActiveInferenceAgentAbsoluteControl
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.agent.heuristic.heuristic_agent import HeuristicAgent
from producer.elasticity.agent.rl.rl_agent import ReinforcementLearningAgent
from producer.elasticity.agent.test.test_agent import TestAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.enums.agent_type import AgentType
from producer.request_handling.request_handler import RequestHandler
from producer.task_generation.task_generator import TaskGenerator


class AgentFactory:
    @staticmethod
    def create(agent_type: AgentType, elasticity_handler: ElasticityHandler, request_handler: RequestHandler,
               task_generator: TaskGenerator, track_slo_stats=True) -> ElasticityAgent | None:
        match agent_type:
            case AgentType.NONE:
                return None
            case AgentType.TEST:
                return TestAgent(elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)
            case AgentType.ACTIVE_INFERENCE:
                return ActiveInferenceAgentRelativeControl(elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)
            case AgentType.HEURISTIC:
                return HeuristicAgent(elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)
            case AgentType.REINFORCEMENT_LEARNING:
                return ReinforcementLearningAgent(elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)
            case AgentType.ACTIVE_INFERENCE_EXPERIMENTAL_1:
                return ActiveInferenceAgentExperimental1(elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)
            case AgentType.ACTIVE_INFERENCE_EXPERIMENTAL_2:
                return ActiveInferenceAgentAbsoluteControl(elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)
            case _:
                return ActiveInferenceAgentRelativeControl(elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)