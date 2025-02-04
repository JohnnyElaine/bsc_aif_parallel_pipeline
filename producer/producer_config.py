from dataclasses import dataclass

from packages.enums import WorkLoad
from packages.enums import WorkType


@dataclass
class ProducerConfig:
    port: int
    video_path: str
    worker_type: WorkType
    work_load: WorkLoad
