import logging
import time
from threading import Thread, Event

from producer.elasticity.agent.agent_factory import AgentFactory
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.enums.agent_type import AgentType
from producer.task_generation.task_generator import TaskGenerator

log = logging.getLogger('producer')


class AgentPipeline(Thread):
    ITERATION_INTERVAL_S = 1

    def __init__(self, elasticity_handler: ElasticityHandler, task_generator: TaskGenerator, agent_type: AgentType, start_task_generator_event: Event):
        super().__init__()
        self._elasticity_handler = elasticity_handler
        self._agent = AgentFactory.create(agent_type, elasticity_handler, task_generator)
        self._start_task_generator_event = start_task_generator_event
        self._is_running = False

    def run(self):
        if self._agent is None:
            return

        log.debug('starting agent-pipeline')
        self._is_running = True
        log.debug("agent-pipeline is waiting for task generator to start")
        self._start_task_generator_event.wait()

        while self._is_running:
            time.sleep(AgentPipeline.ITERATION_INTERVAL_S)
            self._iteration()

        log.debug('stopped agent-pipeline')

    def stop(self):
        self._is_running = False

    def get_slo_statistics(self):
        return self._agent.get_slo_statistics()

    def _iteration(self):
        action, success = self._agent.step()
        log.debug(f'chose action {action}, success: {success}')