from dataclasses import dataclass

from packages.enums import WorkType, LoadingMode, WorkLoad
from producer.enums.agent_type import AgentType


@dataclass
class SimulationConfig:
    producer_ip: str
    producer_port: int
    collector_ip: str
    collector_port: int
    work_type: WorkType
    loading_mode: LoadingMode
    max_work_load: WorkLoad
    agent_type: AgentType
    num_workers: int
    vid_path: str