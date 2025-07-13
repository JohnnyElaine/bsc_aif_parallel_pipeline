import logging
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
from producer.data.worker_knowledge_base import WorkerKnowledgeBase

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
        self._worker_knowledge_base = WorkerKnowledgeBase()
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

    def avg_global_processing_time(self) -> float:
        return self._worker_knowledge_base.avg_global_processing_time()

    def avg_worker_processing_times(self) -> dict[bytes, float]:
        return self._worker_knowledge_base.avg_worker_processing_times()

    def worker_stats_to_df(self) -> pd.DataFrame:
        return self._worker_knowledge_base.stats_to_df()

    def change_inference_quality(self, work_load: InferenceQuality):
        self._inference_quality = work_load
        self._broadcast_change(Task(TaskType.CHANGE_INFERENCE_QUALITY, -1, 0, np.array(work_load.value)))

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
        self._worker_knowledge_base.add_worker(address)

        info = {'type': RepType.REGISTRATION_CONFIRMATION,
                'work_type': self._work_type.value,
                'work_load': self._inference_quality.value,
                'loading_mode': self._loading_mode.value}

        self._channel.send_information(address, info)

    def _handle_work_request(self, address: bytes, req: dict):
        processing_time = req['previous_processing_time']
        self._worker_knowledge_base.add_processing_time(processing_time, address)

        if self._worker_knowledge_base.has_pending_changes(address):
            changes = self._worker_knowledge_base.get_pending_changes(address)
            self._channel.send_information(address, changes)
            return

        self._worker_knowledge_base.increment_stats(address)

        task = self._queue.get() # Task dataclass

        if task.type == RepType.END:
            self._stop_workers(address)
            return

        tasks = [task]

        self._channel.send_work(address, tasks)

    def _handle_first_work_request(self, address: bytes, req: dict):
        self._handle_work_request_function = self._handle_work_request # use regular '_handle_work_request' after
        self.start_task_generation_event.set() # start the task generator when a worker is ready to receive work
        self._handle_work_request(address, req)

    def _broadcast_change(self, change: Task):
        self._worker_knowledge_base.add_change(change)

    def _stop_workers(self, first_worker_addr: bytes):
        # TODO find alternate stopping condition, if not all workers are online

        # stop the initial request
        self._stop_worker(first_worker_addr)

        # stop all remaining workers (n-1)
        for _ in range(self._worker_knowledge_base.size() - 1):
            address, request = self._channel.get_request()
            self._stop_worker(address)

        self._is_running = False

    def _stop_worker(self, address: bytes):
        info = {'type': RepType.END}

        self._channel.send_information(address, info)
        log.debug(f'sent END to {address}')

