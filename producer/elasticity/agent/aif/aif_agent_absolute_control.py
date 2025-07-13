import logging

from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.view.aif_agent_observations import AIFAgentObservations
from producer.request_handling.request_handler import RequestHandler
from producer.task_generation.task_generator import TaskGenerator

log = logging.getLogger('producer')


class ActiveInferenceAgentAbsoluteControl(ElasticityAgent):
    def __init__(self, elasticity_handler: ElasticityHandler, request_handler: RequestHandler, task_generator: TaskGenerator, policy_length: int = 1, track_slo_stats=True):
        super().__init__(elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)

        self.observations = AIFAgentObservations(elasticity_handler.observations(), self.slo_manager)

        # Get the relative actions view for clean interface to increase/decrease actions
        self.actions = elasticity_handler.actions_absolute()

        self.num_resolution_states = len(elasticity_handler.state_resolution.possible_states)
        self.num_fps_states = len(elasticity_handler.state_fps.possible_states)
        self.num_inference_quality_states = len(elasticity_handler.state_inference_quality.possible_states)

    def step(self):
        pass

