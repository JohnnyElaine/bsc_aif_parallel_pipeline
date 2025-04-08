import logging
from queue import Queue

import numpy as np

from packages.data.types.task_type import TaskType
from worker.communication.channel.request_channel import RequestChannel
from worker.work_requesting.work_requester.work_requester import WorkRequester
from packages.data import Task
from packages.network_messages import RepType

log = logging.getLogger('work_requesting')


class ZmqWorkRequester(WorkRequester):
    def __init__(self, shared_task_queue: Queue, request_channel: RequestChannel):
        super().__init__(shared_task_queue)
        self._is_running = False
        self._channel = request_channel

    def run(self):
        log.debug("starting work-requester")
        self._is_running = True

        try:
            ok = True
            while self._is_running and ok:
                ok = self._iteration()

        except EOFError:
            log.info("Producer disconnected. Worker exiting.")

        log.debug("stopped work-requester")

    def stop(self):
        log.info("stopping work-requester")
        self._is_running = False

    def _iteration(self) -> bool:
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
                self._notify_task_processor_of_end()
                return False # Stop the loop by returning False
            case RepType.WORK:
                self._add_tasks_to_queue(tasks)
            case RepType.CHANGE:
                del info['type'] # filter out 'type'. rest of info shows changes
                self._handle_changes(info)
            case _:
                log.debug('received reply of unknown type')

        return True

    def _handle_changes(self, changes: dict):
        for change_type, value in changes.items():
            self._queue.put(Task(change_type, -1, np.array(value)))

    def _notify_task_processor_of_end(self):
        self._queue.put(Task(TaskType.END, -1, np.empty(0)))

    def _add_tasks_to_queue(self, tasks: list[Task]):
        for task in tasks:
            self._queue.put(task)


