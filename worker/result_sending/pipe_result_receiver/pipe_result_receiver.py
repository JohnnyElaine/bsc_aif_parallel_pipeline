import logging
from multiprocessing import Pipe
from threading import Thread
from queue import Queue

log = logging.getLogger('result_sending')


class PipeResultReceiver(Thread):
    def __init__(self, result_queue: Queue, result_pipe: Pipe):
        super().__init__()
        self._queue = result_queue
        self._result_pipe = result_pipe
        self._is_running = False

    def run(self):
        log.debug('starting pipe-result-receiver')
        self._is_running = True

        while self._is_running:
            result = self._result_pipe.recv() # result = Task(id, type, data)
            self._queue.put(result)

    def stop(self):
        self._is_running = False
        log.info('stopped pipe-result-receiver')