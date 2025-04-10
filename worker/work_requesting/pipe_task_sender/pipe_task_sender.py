import logging
from queue import Queue
from threading import Thread

from packages.data.types.task_type import TaskType

log = logging.getLogger('work_requesting')


class PipeTaskSender(Thread):
    def __init__(self, task_queue: Queue, task_pipe):
        super().__init__()
        self._queue = task_queue
        self._task_pipe = task_pipe
        self._is_running = False

    def run(self):
        log.debug('starting pipe-task-sender')
        self._is_running = True

        while self._is_running:
            data = self._queue.get() # LocalMessage Dataclass
            self._send_to_task_processor(data)

            # Indicate that enqueued task is complete and more tasks can be processed. (unblocks .join())
            self._queue.task_done()

            if data.type == TaskType.END:
                self._is_running = False

        log.debug('stopped pipe-task-sender')

    def stop(self):
        log.info('stopping pipe-task-sender')
        self._is_running = False

    def _send_to_task_processor(self, msg):
        self._task_pipe.send(msg)