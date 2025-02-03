from dataclasses import dataclass

from packages.enums import ComputeLoad
from packages.enums import ComputeType


@dataclass
class WorkConfig:
    compute_type: ComputeType
    compute_load: ComputeLoad

