from dataclasses import dataclass

from packages.enums import WorkLoad, WorkType, LoadingMode
from producer.enums.agent_type import AgentType


@dataclass(frozen=True)
class ProducerConfig:
    port: int
    work_type: WorkType
    loading_mode: LoadingMode
    max_work_load: WorkLoad
    agent_type: AgentType
    video_path: str