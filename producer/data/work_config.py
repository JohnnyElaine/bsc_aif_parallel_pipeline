from dataclasses import dataclass

from packages.enums import WorkLoad
from packages.enums import WorkType

@dataclass
class WorkConfig:
    work_type: WorkType
    work_load: WorkLoad