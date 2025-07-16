import time
from dataclasses import asdict

import pandas as pd

from packages.data.local_messages.task import Task
from producer.data.worker_info import WorkerInfo
from producer.data.moving_average import MovingAverage
from producer.statistics.worker_statistics import WorkerStatistics


class WorkerKnowledgeBase:
    def __init__(self):
        self._worker_info_dict = dict()  # key = worker-addr, value = WorkerInfo()
        self._global_processing_time_moving_average = MovingAverage()
        self._worker_statistics_dict = dict()  # key = worker-addr, value = WorkerStatistics()

    def size(self):
        return len(self._worker_info_dict)

    def add_worker(self, address: bytes):
        self._worker_info_dict[address] = WorkerInfo()
        self._worker_statistics_dict[address] = WorkerStatistics(0, time.time())

    def add_processing_time(self, processing_time: float, address:bytes):
        self._global_processing_time_moving_average.add(processing_time)
        self._worker_info_dict[address].add_processing_time(processing_time)

    def has_pending_changes(self, address: bytes) -> bool:
            return self._worker_info_dict[address].has_pending_changes()

    def get_pending_changes(self, address: bytes) -> dict[str, float]:
        return self._worker_info_dict[address].get_all_pending_changes()

    def increment_stats(self, address: bytes):
        self._worker_statistics_dict[address].num_requested_tasks += 1

    def add_change(self, change: Task):
        for worker_addr in self._worker_info_dict.keys():
            self._worker_info_dict[worker_addr].add_change(change)

    def avg_global_processing_time(self) -> float:
        return self._global_processing_time_moving_average.average()

    def avg_worker_processing_times(self) -> dict[bytes, float]:
        return {addr: info.avg_processing_time() for addr, info in self._worker_info_dict.items()}

    def stats_to_df(self) -> pd.DataFrame:
        # Create a nested dictionary with worker addresses as keys
        # and WorkerStatistics asdict() values as inner dictionaries

        # only use id of worker-addr so the index can be an integer
        data_dict = {int(addr.decode('utf-8').split('-')[1]): asdict(stats) for addr, stats in self._worker_statistics_dict.items()}

        # Convert to DataFrame using from_dict with orient='index' to make worker worker_addr the index
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df.index.name = 'worker_id'

        return df.sort_index()