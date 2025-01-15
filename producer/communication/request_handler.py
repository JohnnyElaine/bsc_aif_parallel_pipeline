import logging
from threading import Thread
from queue import Queue

from producer.communication.channel.router_channel import RouterChannel

log = logging.getLogger("producer")


class RequestHandler(Thread):
    def __init__(self, port: int, shared_queue: Queue):
        super().__init__()
        self._channel = RouterChannel(port)
        self._queue = shared_queue
        self._is_running = False


    def run(self):
        self._is_running = True
        self._channel.bind()

        while self._is_running:
            task = self._queue.get()

            address, request = self._channel.get_request()

            tasks = [task]

            self._channel.send_work(address, tasks)