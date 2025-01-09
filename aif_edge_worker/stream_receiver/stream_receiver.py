from abc import ABC
from threading import Thread
from queue import Queue

class StreamReceiver(ABC, Thread):
    def __init__(self, shared_queue: Queue):
        super().__init__()
        self._queue = shared_queue
