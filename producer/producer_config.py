from dataclasses import dataclass

from packages.enums import ComputeLoad
from packages.enums import ComputeType


@dataclass
class ProducerConfig:
    port: int
    video_path: str
    compute_type: ComputeType
    compute_load: ComputeLoad
