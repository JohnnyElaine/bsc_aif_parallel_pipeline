from dataclasses import dataclass

from aif_edge_worker.enums.computation_workload import ComputationWorkload
from aif_edge_worker.enums.computation_type import ComputationType
from aif_edge_worker.enums.stream_source import StreamSource


@dataclass
class StreamConfig:
    fps: int
    resolution: tuple[int, int]
    computation_type: ComputationType
    computation_workload: ComputationWorkload
    stream_source: StreamSource

