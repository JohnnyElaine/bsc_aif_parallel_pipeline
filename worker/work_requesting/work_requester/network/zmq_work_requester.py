import logging
import time
from multiprocessing import Value
from queue import Queue

import numpy as np

from packages.data import Task
from packages.data.local_messages.task_type import TaskType
from packages.network_messages import RepType
from worker.communication.channel.request_channel import RequestChannel
from worker.work_requesting.work_requester.work_requester import WorkRequester

log = logging.getLogger('work_requesting')


class ZmqWorkRequester(WorkRequester):
    def __init__(self, shared_task_queue: Queue, request_channel: RequestChannel, shared_processing_time: Value, outage_config=None):
        super().__init__(shared_task_queue)
        self._is_running = False
        self._channel = request_channel
        self._latest_processing_time = shared_processing_time
        self._outage_config = outage_config
        self._req_work_function = self._req_work if outage_config is None else self._req_work_with_outage_config
        self._num_requested_tasks = 0
        self._start_time = 0

    def run(self):
        log.debug('starting work-requester')
        self._is_running = True
        self._start_time = time.perf_counter()

        try:
            ok = True
            while self._is_running and ok:
                ok = self._iteration()

        except EOFError:
            log.info('Producer disconnected. Worker exiting.')

        log.debug('stopped work-requester')

    def _iteration(self) -> bool:
        """
        :return: True if the iteration was successful. False otherwise.
        """
        self._queue.join() # Block until queue is empty

        info, tasks = self._req_work_function()

        return self._handle_work(info, tasks)

    def _handle_work(self, info: dict, tasks: list) -> bool:
        rep_type = info['type']

        match rep_type:
            case RepType.WORK:
                self._add_tasks_to_queue(tasks)
            case RepType.CHANGE:
                del info['type'] # filter out 'type', rest of info shows changes
                self._handle_changes(info)
            case RepType.END:
                log.debug('received END')
                self._notify_task_processor_of_end()
                return False  # Stop the loop by returning False
            case _:
                log.debug('received reply of unknown type')

        return True

    def _handle_changes(self, changes: dict):
        for change_type, value in changes.items():
            self._queue.put(Task(change_type, -1, 0, np.array(value)))

    def _notify_task_processor_of_end(self):
        self._queue.put(Task(TaskType.END, -1, 0, np.empty(0)))

    def _add_tasks_to_queue(self, tasks: list[Task]):
        for task in tasks:
            self._queue.put(task)

    def _req_work(self) -> tuple[dict, list[Task]]:
        # Get the latest processing time from shared memory
        with self._latest_processing_time.get_lock():
            processing_time = self._latest_processing_time.value
        
        return self._channel.get_work(processing_time)

    def _req_work_with_outage_config(self) -> tuple[dict, list[Task]]:
        time_since_start = time.perf_counter() - self._start_time
        if time_since_start >= self._outage_config.time_until_outage:
            log.info(f'After {time_since_start} seconds: Simulating offline worker for {self._outage_config.duration} seconds')
            time.sleep(self._outage_config.duration)
            log.info(f'Worker back online')

        return self._req_work()

