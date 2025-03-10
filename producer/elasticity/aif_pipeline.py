import time
import logging
from threading import Thread, Event

from producer.elasticity.handler.elasticity_handler import ElasticityHandler

log = logging.getLogger('producer')


class ActiveInferencePipeline(Thread):
    TIME_INTERVAL_S = 1
    def __init__(self, elasticity_handler: ElasticityHandler, start_task_generator_event: Event):
        super().__init__()
        self._elasticity_handler = elasticity_handler
        self._start_task_generator_event = start_task_generator_event
        self._is_running = False

    def run(self):
        self._is_running = True
        log.debug("waiting for task generator to start")
        self._start_task_generator_event.wait()

        while self._is_running:
            time.sleep(ActiveInferencePipeline.TIME_INTERVAL_S)
            self._check_all_slo()

    def stop(self):
        self._is_running = False

    def _check_all_slo(self):
        #self._elasticity_handler.change_work_load(WorkLoad.LOW)
        self.stop()