from dataclasses import dataclass

from worker.enums.compute_load import ComputeLoad
from worker.enums.compute_type import ComputeType
from worker.enums.stream_source import StreamSource


@dataclass
class PullStreamConfig:
    fps: int
    resolution: tuple[int, int]
    compute_type: ComputeType
    compute_load: ComputeLoad
    stream_source: StreamSource

