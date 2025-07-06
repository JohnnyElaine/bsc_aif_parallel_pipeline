import logging
import time
from dataclasses import asdict
from queue import Queue
from threading import Thread, Event

import numpy as np
import pandas as pd

from packages.data.local_messages.task import Task
from packages.data.local_messages.task_type import TaskType
from packages.enums import LoadingMode
from packages.enums import WorkType, InferenceQuality
from packages.network_messages import ReqType, RepType
from producer.communication.channel.router_channel import RouterChannel
from producer.data.worker_info import WorkerInfo
from producer.statistics.moving_average import MovingAverage
from producer.statistics.worker_statistics import WorkerStatistics

log = logging.getLogger('producer')


class RequestHandler(Thread):
    def __init__(self, port: int, shared_queue: Queue, work_type: WorkType, inference_quality: InferenceQuality,
                 loading_mode: LoadingMode, start_task_generation_event: Event):
        super().__init__()
        self._channel = RouterChannel(port)
        self._queue = shared_queue
        self._work_type = work_type
        self._inference_quality = inference_quality
        self._loading_mode = loading_mode
        self._worker_knowledge_base = dict() # key = worker-addr, value = WorkerInfo()
        self._worker_statistics = dict() # key = worker-addr, value = WorkerStatistics()
        self._global_processing_time_moving_average = MovingAverage()
        self._is_running = False

        self._handle_work_request_function = self._handle_first_work_request
        self.start_task_generation_event = start_task_generation_event

    def run(self):
        self._is_running = True
        self._channel.bind()
        log.debug(f'bound {self._channel}')

        while self._is_running:
            address, request = self._channel.get_request()
            self._handle_request(address, request)

        self._channel.close()
        log.debug('stopped request-handler')

    def stop(self):
        self._is_running = False
        self._channel.close()

    def change_inference_quality(self, work_load: InferenceQuality):
        self._inference_quality = work_load
        self._broadcast_change(Task(TaskType.CHANGE_INFERENCE_QUALITY, -1, np.array(work_load.value)))

    def get_worker_statistics(self) -> pd.DataFrame:
        # Create a nested dictionary with worker addresses as keys
        # and WorkerStatistics asdict() values as inner dictionaries

        # only use id of worker-addr so the index can be an integer
        data_dict = {int(addr.decode('utf-8').split('-')[1]): asdict(stats) for addr, stats in self._worker_statistics.items()}

        # Convert to DataFrame using from_dict with orient='index' to make worker worker_addr the index
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df.index.name = 'worker_id'

        return df.sort_index()

    def _handle_request(self, address: bytes, request: dict) -> bool:
        req_type = request['type']

        match req_type:
            case ReqType.REGISTER:
                self._handle_register_request(address)
            case ReqType.GET_WORK:
                self._handle_work_request_function(address, request)
            case _:
                log.info(f"Received unknown request type: {req_type}")

        return True

    def _handle_register_request(self, address: bytes):
        self._worker_knowledge_base[address] = WorkerInfo()
        self._worker_statistics[address] = WorkerStatistics(0, time.time())

        info = {'type': RepType.REGISTRATION_CONFIRMATION,
                'work_type': self._work_type.value,
                'work_load': self._inference_quality.value,
                'loading_mode': self._loading_mode.value}

        self._channel.send_information(address, info)

    def _handle_work_request(self, address: bytes, req: dict):
        processing_time = req['previous_processing_time']
        self._global_processing_time_moving_average.add(processing_time)
        self._worker_knowledge_base[address].add_processing_time(processing_time)

        if self._worker_knowledge_base[address].has_pending_changes():
            changes = self._worker_knowledge_base[address].get_all_pending_changes()
            self._channel.send_information(address, changes)
            return

        self._worker_statistics[address].num_requested_tasks += 1

        task = self._queue.get() # Task dataclass

        if task.type == RepType.END:
            self._stop_workers(address)
            return

        tasks = [task]

        self._channel.send_work(address, tasks)

    def _handle_first_work_request(self, address: bytes, req:dict):
        self._handle_work_request_function = self._handle_work_request # use regular '_handle_work_request' after
        self.start_task_generation_event.set() # start the task generator when a worker is ready to receive work
        self._handle_work_request(address, req)

    def _broadcast_change(self, change: Task):
        for worker_addr in self._worker_knowledge_base.keys():
            self._worker_knowledge_base[worker_addr].add_change(change)

    def _stop_workers(self, first_worker_addr: bytes):
        # TODO find alternate stopping condition, if not all workers are online

        # stop the initial request
        self._stop_worker(first_worker_addr)

        # stop all remaining workers (n-1)
        for _ in range(len(self._worker_knowledge_base) - 1):
            address, request = self._channel.get_request()
            self._stop_worker(address)

        self._is_running = False

    def _stop_worker(self, address: bytes):
        self._worker_knowledge_base[address].has_received_end_message = True

        info = {'type': RepType.END}

        self._channel.send_information(address, info)
        log.debug(f'sent END to {address}')

