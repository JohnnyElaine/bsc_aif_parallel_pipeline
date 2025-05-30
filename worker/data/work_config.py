from dataclasses import dataclass

from packages.enums import InferenceQuality, WorkType, LoadingMode


@dataclass
class WorkConfig:
    work_type: WorkType
    inference_quality: InferenceQuality
    loading_mode: LoadingMode