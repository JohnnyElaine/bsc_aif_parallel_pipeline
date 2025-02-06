import logging
from queue import Queue

import numpy as np

from worker.communication.channel.request_channel import RequestChannel
from worker.communication.work_requester.work_requester import WorkRequester
from packages.data import Task
from packages.message_types import RepType

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

        info, tasks = self._channel.get_work()

        return self._handle_work(info, tasks)

    def _handle_work(self, info: dict, tasks: list) -> bool:
        rep_type = info['type']

        match rep_type:
            case RepType.END:
                return False
            case RepType.WORK:
                pass # TODO do something idk
            case RepType.WORK_AND_CHANGE:
                del info['type'] # filter out 'type'. rest of info shows changes
                self._handle_changes(info)
            case _:
                pass # do nothing

        self._add_to_queue(tasks)

        return True

    def _handle_changes(self, changes: dict):
        for change_type, value in changes.items():
            self._handle_work(change_type, value)

    def _handle_change(self, change_type: str, value):
        match change_type:
            case 'work_load':
                self._change_work_load(value)
            case _:
                pass

    def _change_work_load(self, value):
        pass # TODO

    def _add_to_queue(self, tasks: list[Task]):
        for task in tasks:
            self._queue.put(task)

    @staticmethod
    def reconstruct_task(task_buffered, dtype, shape):
        task = np.frombuffer(task_buffered, dtype=dtype)
        return task.reshape(shape)
