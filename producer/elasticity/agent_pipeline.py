import logging
import time
from threading import Thread

from producer.elasticity.agent.agent_factory import AgentFactory
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.enums.agent_type import AgentType
from producer.request_handling.request_handler import RequestHandler
from producer.task_generation.task_generator import TaskGenerator

log = logging.getLogger('producer')


class AgentPipeline(Thread):
    ITERATION_INTERVAL_S = 0.5

    def __init__(self, elasticity_handler: ElasticityHandler, task_generator: TaskGenerator, request_handler: RequestHandler, agent_type: AgentType, track_slo_stats=True):
        super().__init__()
        self._elasticity_handler = elasticity_handler
        self._agent = AgentFactory.create(agent_type, elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)
        self._start_task_generator_event = request_handler.start_task_generation_event
        self._is_running = False

    def run(self):
        if self._agent is None:
            return

        log.debug('starting agent-pipeline')
        self._is_running = True
        log.debug("agent-pipeline is waiting for task generator to start")
        self._start_task_generator_event.wait()
        # TODO add a start delay so the aif agent does not learn from the random data at the start
        log.debug('agent-pipeline active')

        while self._is_running:
            time.sleep(AgentPipeline.ITERATION_INTERVAL_S)
            self._iteration()

        log.debug('stopped agent-pipeline')

    def stop(self):
        self._is_running = False

    def get_slo_statistics(self):
        return self._agent.get_slo_statistics()

    def _iteration(self):
        self._agent.step()