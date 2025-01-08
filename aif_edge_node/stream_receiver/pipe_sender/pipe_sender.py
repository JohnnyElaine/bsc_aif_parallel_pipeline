import logging
from threading import Thread
from queue import Queue

log = logging.getLogger("node")


class PipeSender(Thread):
    def __init__(self, shared_queue: Queue, pipe_to_stream_computer, ):
        super().__init__()
        self._queue = shared_queue
        self._pipe_to_stream_computer = pipe_to_stream_computer
        self._is_running = False

    def run(self):
        log.debug('starting pipe-sender')
        self._is_running = True

        while self._is_running:
            data = self._queue.get()
            self._send_to_pipe(data)


    def stop(self):
        self._is_running = False


    def _send_to_pipe(self, msg):
        self._pipe_to_stream_computer.send(msg)


