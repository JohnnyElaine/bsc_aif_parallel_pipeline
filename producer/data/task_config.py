from dataclasses import dataclass

from packages.enums import WorkLoad
from packages.enums import WorkType
from producer.data.resolution import Resolution


@dataclass
class TaskConfig:
    work_type: WorkType
    max_work_load: WorkLoad
    max_resolution: Resolution
    max_fps: int

