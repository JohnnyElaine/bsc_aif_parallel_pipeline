
from producer.elasticity.agent.action.action_type import ActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler


class TestAgent(ElasticityAgent):


    def __init__(self, elasticity_handler: ElasticityHandler):
        super().__init__(elasticity_handler)
        self.count = 0


    def step(self) -> tuple[ActionType, bool]:
        """
        Perform a single step of the test Agent
        Returns:
            tuple[ActionType, bool]: The action taken and whether it was successful
        """
        self.slo_manager.get_all_slo_status(track_statistics=True)

        self.count += 1
        if self.count == -1:
            success = self.elasticity_handler.decrease_work_load()
            return ActionType.DECREASE_WORK_LOAD, success

        return ActionType.DO_NOTHING, True
