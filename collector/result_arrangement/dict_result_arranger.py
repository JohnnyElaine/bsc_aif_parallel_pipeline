import logging
import time
from threading import Thread
from queue import Queue

from collector.datastructures.blocking_dict import BlockingDict
from packages.data import Task

log = logging.getLogger('collector')


class DictResultArranger(Thread):
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

    def stop(self):
        self._is_running = False

    def _iteration(self) -> bool:
        result = self._result_dict.pop(self._expected_id)

        if result is None:
            self._handle_missing_task()
            return True

        # skip past frames
        if result.id < self._expected_id:
            return True

        self._handle_expected_task(result)

        return True

    def _handle_missing_task(self):
        self._strikes += 1
        if self._strikes >= DictResultArranger.MAX_STRIKES:
            log.debug(f'skipped task with id={self._expected_id}')
            self._strikes = 0
            self._expected_id += 1 # skipping task
        else:
            # expected task might be available later
            time.sleep(DictResultArranger.WAIT_TIME)

    def _handle_expected_task(self, result: Task):
        self._output_queue.put(result.data)
        self._strikes = 0
        self._expected_id += 1

