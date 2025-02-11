from dataclasses import dataclass

from packages.enums import WorkLoad, WorkType, LoadingMode


@dataclass(frozen=True)
class ProducerConfig:
    port: int
    video_path: str
    work_type: WorkType
    work_load: WorkLoad
    loading_mode: LoadingMode
