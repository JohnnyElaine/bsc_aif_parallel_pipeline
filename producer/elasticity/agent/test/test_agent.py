
from producer.elasticity.action.general_action_type import GeneralActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.request_handling.request_handler import RequestHandler
from producer.task_generation.task_generator import TaskGenerator


class TestAgent(ElasticityAgent):


    def __init__(self, elasticity_handler: ElasticityHandler, request_handler: RequestHandler, task_generator: TaskGenerator, track_slo_stats=True):
        super().__init__(elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)
        self.count = 0

    def step(self) -> tuple[GeneralActionType, bool]:
        """
        Perform a single step of the test Agent
        Returns:
            tuple[GeneralActionType, bool]: The action taken and whether it was successful
        """
        self.slo_manager.get_all_slo_status(track_stats=True)

        self.count += 1
        if self.count == 5:
            success = self.elasticity_handler.decrease_inference_quality()
            return GeneralActionType.DECREASE_INFERENCE_QUALITY, success

        #if self.count % 2 == 0:
        #    return GeneralActionType.DECREASE_FPS, self.elasticity_handler.decrease_fps()

        return GeneralActionType.NONE, True
