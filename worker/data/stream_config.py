from dataclasses import dataclass

from worker.enums.compute_load import ComputeLoad
from worker.enums.compute_type import ComputeType


@dataclass
class StreamConfig:
    compute_type: ComputeType
    compute_load: ComputeLoad

