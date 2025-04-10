import logging
import time
from threading import Thread
from queue import Queue

import numpy as np

from collector.constants.constants import END_TASK_ID
from collector.datastructures.blocking_dict import BlockingDict
from packages.data import Task

log = logging.getLogger('collector')


class ResultMapper(Thread):
    WAIT_TIME = int(1/30) # TODO find more suitable wait time
    MAX_STRIKES = 3
    def __init__(self, result_dict: BlockingDict, output_queue: Queue):
        super().__init__()
        self._result_dict = result_dict # key = result_id, val = result
        self._output_queue = output_queue
        self._is_running = False

        self._expected_id = 0
        self._strikes = 0

    def run(self):
        self._is_running = True
        ok = True

        while self._is_running and ok:
            ok = self._iteration()

        log.debug('stopped result-mapper')

    def stop(self):
        self._is_running = False

    def _iteration(self) -> bool:
        result = self._result_dict.pop(self._expected_id)

        if END_TASK_ID in self._result_dict:
            return False

        if result is None:
            self._handle_missing_task()
            return True

        # skip past frames that arrived too late
        if result.id < self._expected_id:
            return True

        self._handle_expected_task(result)

        return True

    def _handle_missing_task(self):
        self._strikes += 1
        if self._strikes >= ResultMapper.MAX_STRIKES:
            log.debug(f'skipped task with id={self._expected_id}')
            self._strikes = 0
            self._expected_id += 1 # skipping task
        else:
            # expected task might be available later
            time.sleep(ResultMapper.WAIT_TIME)

    def _handle_expected_task(self, result: Task):
        self._output_queue.put(result.data)
        self._strikes = 0
        self._expected_id += 1

    def _stop_output_viewer(self):
        self._output_queue.put(np.array(-1))

