import logging
from threading import Thread
from queue import Queue

log = logging.getLogger("worker")


class PipeSender(Thread):
    def __init__(self, shared_task_queue: Queue, pipe_to_task_processor):
        super().__init__()
        self._queue = shared_task_queue
        self._pipe_to_task_processor = pipe_to_task_processor
        self._is_running = False

    def run(self):
        log.debug('starting pipe-sender')
        self._is_running = True

        while self._is_running:
            data = self._queue.get()
            self._send_to_pipe(data)

            # Indicate that enqueued task is complete and more tasks can be processed. (unblocks .join())
            self._queue.task_done()

    def stop(self):
        log.info('stopping pipe-sender')
        self._is_running = False

    def _send_to_pipe(self, msg):
        self._pipe_to_task_processor.send(msg)