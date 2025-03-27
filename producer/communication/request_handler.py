import logging
import time
from threading import Thread, Event
from queue import Queue

from packages.data import Instruction, InstructionType
from packages.enums import WorkType, WorkLoad
from packages.message_types import ReqType, RepType
from producer.communication.channel.router_channel import RouterChannel
from producer.data.worker_info import WorkerInfo
from packages.enums import LoadingMode
from producer.statistics.worker_statistics import WorkerStatistics

log = logging.getLogger("producer")


class RequestHandler(Thread):
    def __init__(self, port: int, shared_queue: Queue, work_type: WorkType, work_load: WorkLoad, loading_mode: LoadingMode):
        super().__init__()
        self._channel = RouterChannel(port)
        self._queue = shared_queue
        self._work_type = work_type
        self._work_load = work_load
        self._loading_mode = loading_mode
        self._worker_knowledge_base = dict() # key = worker-addr, value = WorkerInfo()
        self._worker_statistics = dict() # key = worker-addr, value = WorkerStatistics()
        self._is_running = False

        self._current_work_request_handler = self._handle_first_work_request
        self.start_task_generator_event = Event()

    def run(self):
        self._is_running = True
        self._channel.bind()
        log.debug(f'bound {self._channel}')

        while self._is_running:
            address, request = self._channel.get_request()

            self._handle_request(address, request)

    def stop(self):
        self._is_running = False
        self._channel.close()

    def change_work_load(self, work_load: WorkLoad):
        self._work_load = work_load
        self._broadcast_instruction(Instruction(InstructionType.CHANGE_WORK_LOAD, work_load.value))

    def _handle_request(self, address: bytes, request: dict):
        req_type = request['type']

        match req_type:
            case ReqType.REGISTER:
                self._handle_register_request(address)
            case ReqType.GET_WORK:
                self._current_work_request_handler(address)
            case _:
                log.debug(f"Received unknown request type: {req_type}")

    def _handle_register_request(self, address: bytes):
        self._worker_knowledge_base[address] = WorkerInfo()
        self._worker_statistics[address] = WorkerStatistics(0, time.time())

        info = {'type': RepType.REGISTRATION_CONFIRMATION,
                'work_type': self._work_type.value,
                'work_load': self._work_load.value,
                'loading_mode': self._loading_mode.value}

        self._channel.send_information(address, info)

    def _handle_work_request(self, address: bytes):
        if self._worker_knowledge_base[address].has_pending_instructions():
            instruction = self._worker_knowledge_base[address].get_all_pending_instructions()
            self._channel.send_information(address, instruction)
            return

        self._worker_statistics[address].num_requested_tasks += 1

        task = self._queue.get()

        # TODO Find a way to send multiple tasks (if available) at once should the worker info prefer it.
        tasks = [task]

        self._channel.send_work(address, tasks)

    def _handle_first_work_request(self, address: bytes):
        self._current_work_request_handler = self._handle_work_request # use regular '_handle_work_request' after
        self.start_task_generator_event.set() # start the task generator when a worker is ready
        self._handle_work_request(address)

    def _broadcast_instruction(self, instruction: Instruction):
        for worker_addr in self._worker_knowledge_base.keys():
            self._worker_knowledge_base[worker_addr].add_instruction(instruction)

