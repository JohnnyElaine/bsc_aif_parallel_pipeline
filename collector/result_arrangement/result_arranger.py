import logging
import time
from queue import PriorityQueue
from threading import Thread
from queue import Queue

from packages.data import Task

log = logging.getLogger('collector')


class ResultArranger(Thread):
    WAIT_TIME = int(1/30) # TODO find more suitable wait time
    MAX_STRIKES = 3
    def __init__(self, result_queue: PriorityQueue, output_queue: Queue):
        super().__init__()
        self._result_queue = result_queue
        self._output_queue = output_queue
        self._is_running = False

        self._buffer = dict()  # key = result_id, val = result
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
        result = self._result_queue.get()

        # skip past frames
        if result.id < self._expected_id:
            return True

        self._buffer[result.id] = result

        if self._expected_id not in self._buffer:
            self._handle_missing_task()
            return True

        self._handle_expected_task(result)

        return True

    def _handle_missing_task(self):
        self._strikes += 1
        if self._strikes >= ResultArranger.MAX_STRIKES:
            log.debug(f'skipped task with id={self._expected_id}')
            self._strikes = 0
            self._expected_id += 1 # skipping task
        else:
            # expected task might be available later
            time.sleep(ResultArranger.WAIT_TIME)

    def _handle_expected_task(self, result: Task):
        self._output_queue.put(result.data)
        self._strikes = 0
        self._expected_id += 1

