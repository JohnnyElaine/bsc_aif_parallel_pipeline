import logging
from threading import Thread, Event
from queue import Queue

from packages.message_types import ReqType, RepType
from producer.communication.channel.router_channel import RouterChannel
from producer.data.work_config import WorkConfig

log = logging.getLogger("producer")


class RequestHandler(Thread):
    def __init__(self, port: int, shared_queue: Queue, initial_work_config: WorkConfig):
        super().__init__()
        self._channel = RouterChannel(port)
        self._queue = shared_queue
        self._work_config = initial_work_config
        self._is_running = False

        self._work_request = self._handle_first_work_request

        self.worker_ready_event = Event()

    def run(self):
        self._is_running = True
        self._channel.bind()
        log.debug(f'bound {self._channel}')

        while self._is_running:
            address, request = self._channel.get_request()

            self._handle_request(address, request)

    def shutdown(self):
        self._is_running = False
        self._channel.close()

    def _handle_request(self, address: bytes, request: dict):
        req_type = request['type']

        match req_type:
            case ReqType.REGISTER:
                self._handle_register_request(address)
            case ReqType.GET_WORK:
                self._work_request(address)
            case _:
                log.debug(f"Received unknown request type: {req_type}")

    def _handle_register_request(self, address: bytes):
        info = {'type': RepType.REGISTRATION_CONFIRMATION,
                'work_type': self._work_config.work_type.value,
                'work_load': self._work_config.work_load.value}

        self._channel.send_information(address, info)

    def _handle_work_request(self, address: bytes):
        # start the task generator when a worker is ready
        self.worker_ready_event.set()

        task = self._queue.get()

        # TODO Why array? potentially send multiple tasks at once
        tasks = [task]

        self._channel.send_work(address, tasks)

    def _handle_first_work_request(self, address: bytes):
        # use '_handle_work_request' instead of this function for all other requests
        self._work_request = self._handle_work_request

        # start the task generator when a worker is ready
        self.worker_ready_event.set()

        self._handle_work_request(address)

