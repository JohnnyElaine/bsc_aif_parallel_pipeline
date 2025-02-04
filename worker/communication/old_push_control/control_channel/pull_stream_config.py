from dataclasses import dataclass

from packages.enums import WorkLoad
from packages.enums import WorkType
from packages.enums import WorkType
from worker.enums.work_source import WorkSource


@dataclass
class PullStreamConfig:
    fps: int
    resolution: tuple[int, int]
    work_type: WorkType
    work_load: WorkLoad
    stream_source: WorkSource

