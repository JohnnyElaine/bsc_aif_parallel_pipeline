from dataclasses import dataclass

from packages.enums import WorkLoad, WorkType, LoadingMode


@dataclass
class WorkConfig:
    work_type: WorkType
    work_load: WorkLoad
    loading_mode: LoadingMode