import logging
from queue import Queue

import numpy as np

from worker.communication.channel.request_channel import RequestChannel
from worker.communication.work_requester import work_type
from worker.communication.work_requester.work_requester import WorkRequester
from worker.data.task import Task

log = logging.getLogger("worker")


class ZmqWorkRequester(WorkRequester):
    def __init__(self, shared_queue: Queue, request_channel: RequestChannel):
        super().__init__(shared_queue)
        self._is_running = True
        self._channel = request_channel

    def run(self):
        log.debug("starting stream-receiver")
        self._is_running = True

        try:
            while self._is_running:
                ok = self._iteration()
                if not ok:
                    self.stop()
        except EOFError:
            log.info("Producer disconnected. Worker exiting.")
            self.stop()

    def stop(self):
        log.info("stopping stream-receiver")
        self._is_running = False

    def _iteration(self):
        """
        :return: True if the iteration was successful. False otherwise.
        """

        self._queue.join() # Block until queue is empty

        work = self._channel.get_work()

        if work is None:
            return False

        info, tasks = work

        self._handle_work(info, tasks)

        return True

    def _handle_work(self, info: dict, tasks: list):
        type = info["type"]

        match type:
            case work_type.COMPUTE:
                pass # TODO do something idk
            case work_type.COMPUTE_AND_CHANGE_COMPUTE_TYPE:
                pass
                # TODO: change compute type
            case _:
                pass

        self._add_to_queue(tasks)

    def _add_to_queue(self, tasks: list[Task]):
        for task in tasks:
            self._queue.put(task)

    @staticmethod
    def reconstruct_task(task_buffered, dtype, shape):
        task = np.frombuffer(task_buffered, dtype=dtype)
        return task.reshape(shape)
