from enum import Enum


class InferenceQuality(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    @staticmethod
    def int_to_enum(value: int):
        match value:
            case InferenceQuality.LOW.value:
                return InferenceQuality.LOW
            case InferenceQuality.MEDIUM.value:
                return InferenceQuality.MEDIUM
            case InferenceQuality.HIGH.value:
                return InferenceQuality.HIGH
            case _:
                return InferenceQuality.MEDIUM