from abc import ABC, abstractmethod

from packages.enums import WorkType, LoadingMode, WorkLoad
from producer.enums.agent_type import AgentType
from worker.worker import Worker
from worker.worker_config import WorkerConfig


class Simulation(ABC):
    def __init__(self, producer_ip: str,
                 producer_port: int,
                 collector_ip: str,
                 collector_port: int,
                 work_type: WorkType,
                 loading_mode: LoadingMode,
                 max_work_load: WorkLoad,
                 agent_type: AgentType,
                 worker_capacities: list[float],
                 vid_path: str):
        self.producer_ip = producer_ip
        self.producer_port = producer_port
        self.collector_ip = collector_ip
        self.collector_port = collector_port
        self.work_type = work_type
        self.loading_mode = loading_mode
        self.max_work_load = max_work_load
        self.agent_type = agent_type
        self.worker_capacities = worker_capacities
        self.vid_path = vid_path

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def create_workers(processing_delays: list[float], producer_ip: str, producer_port: int, collector_ip: str, collector_port: int):
        return [Worker(WorkerConfig(i, producer_ip, producer_port, collector_ip, collector_port, delay)) for i, delay in enumerate(processing_delays)]