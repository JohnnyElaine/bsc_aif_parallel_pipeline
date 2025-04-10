from abc import ABC
from queue import Queue
from threading import Thread


class WorkRequester(ABC, Thread):
    def __init__(self, shared_queue: Queue):
        super().__init__()
        self._queue = shared_queue
