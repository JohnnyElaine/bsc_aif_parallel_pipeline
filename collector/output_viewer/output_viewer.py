import logging
import time
import cv2 as cv
from threading import Thread
from queue import Queue
from numpy import ndarray

import packages.time_util as time_util
from collector.constants.constants import END_TASK_ID

log = logging.getLogger('collector')


class OutputViewer(Thread):
    def __init__(self, output_queue: Queue):
        super().__init__()
        self._output_queue = output_queue
        self._target_frame_time = 1 / 30 # TODO communicate with producer to get current output fps
        self._is_running = False

    def run(self):
        self._is_running = True
        ok = True
        while self._is_running and ok:
            ok = self._iteration()

        log.debug('stopped output-viewer')

    def _iteration(self) -> bool:
        iteration_start_time = time.perf_counter()
        frame = self._output_queue.get()

        if frame == END_TASK_ID:
            return False

        if not OutputViewer._display_frame(frame):
            return False

        time_util.enforce_target_fps(iteration_start_time, self._target_frame_time)

        return True

    @staticmethod
    def _display_frame(frame: ndarray) -> bool:
        cv.imshow('Collector', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return False

        return True


