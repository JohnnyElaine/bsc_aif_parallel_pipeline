import logging
from queue import Queue

from worker.communication.channel.request_channel import RequestChannel
from worker.work_requesting.work_requester.work_requester import WorkRequester
from packages.data import Task, Instruction
from packages.message_types import RepType

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
            while self._is_running:
                ok = self._iteration()
                if not ok:
                    self.stop()
                    break
                    
        except EOFError:
            log.info("Producer disconnected. Worker exiting.")
            self.stop()

    def stop(self):
        log.info("stopping work-requester")
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
                self._add_tasks_to_queue(tasks)
            case RepType.INSTRUCTION:
                del info['type'] # filter out 'type'. rest of info shows changes
                self._handle_instructions(info)
            case _:
                pass # do nothing

        return True

    def _handle_instructions(self, changes: dict):
        for instruction_type, value in changes.items():
            self._queue.put(Instruction(instruction_type, value))

    def _add_tasks_to_queue(self, tasks: list[Task]):
        for task in tasks:
            self._queue.put(task)