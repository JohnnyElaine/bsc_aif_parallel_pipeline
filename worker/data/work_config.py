from dataclasses import dataclass

from packages.enums import WorkLoad
from packages.enums import WorkType
from worker.enums.loading_mode import LoadingMode


@dataclass
class WorkConfig:
    work_type: WorkType
    work_load: WorkLoad
    loading_mode: LoadingMode

