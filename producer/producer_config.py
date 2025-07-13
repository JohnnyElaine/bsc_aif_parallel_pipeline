from dataclasses import dataclass

from packages.enums import InferenceQuality, WorkType, LoadingMode
from producer.data.stream_multiplier_entry import StreamMultiplierEntry
from producer.enums.agent_type import AgentType


@dataclass(frozen=True)
class ProducerConfig:
    port: int
    work_type: WorkType
    loading_mode: LoadingMode
    max_inference_quality: InferenceQuality
    agent_type: AgentType
    video_path: str
    track_slo_stats: bool = True
    initial_stream_multiplier: int = 1
    # Variable computational demand: list of StreamMultiplierEntry objects
    stream_multiplier_schedule: list[StreamMultiplierEntry] = None