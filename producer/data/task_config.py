from dataclasses import dataclass

from packages.enums import InferenceQuality
from packages.enums import WorkType
from producer.data.resolution import Resolution


@dataclass
class TaskConfig:
    work_type: WorkType
    max_work_load: InferenceQuality
    max_resolution: Resolution
    max_fps: int

