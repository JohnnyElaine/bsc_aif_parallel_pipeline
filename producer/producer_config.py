from dataclasses import dataclass

from packages.enums import InferenceQuality, WorkType, LoadingMode
from producer.enums.agent_type import AgentType


@dataclass(frozen=True)
class ProducerConfig:
    port: int
    work_type: WorkType
    loading_mode: LoadingMode
    max_inference_quality: InferenceQuality
    agent_type: AgentType
    video_path: str