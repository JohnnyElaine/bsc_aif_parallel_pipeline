from abc import ABC, abstractmethod

from packages.enums import WorkType, LoadingMode, InferenceQuality
from producer.enums.agent_type import AgentType


class Simulation(ABC):
    def __init__(self,
                 producer_ip: str,
                 producer_port: int,
                 collector_ip: str,
                 collector_port: int,
                 work_type: WorkType,
                 loading_mode: LoadingMode,
                 max_work_load: InferenceQuality,
                 agent_type: AgentType,
                 vid_path: str):
        self.producer_ip = producer_ip
        self.producer_port = producer_port
        self.collector_ip = collector_ip
        self.collector_port = collector_port
        self.work_type = work_type
        self.loading_mode = loading_mode
        self.max_work_load = max_work_load
        self.agent_type = agent_type
        self.vid_path = vid_path

    @abstractmethod
    def run(self):
        pass