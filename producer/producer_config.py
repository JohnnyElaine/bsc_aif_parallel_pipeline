from dataclasses import dataclass

from packages.enums import WorkLoad
from packages.enums import WorkType
from worker.enums.loading_mode import LoadingMode


@dataclass(frozen=True)
class ProducerConfig:
    port: int
    video_path: str
    work_type: WorkType
    work_load: WorkLoad
    loading_mode: LoadingMode
