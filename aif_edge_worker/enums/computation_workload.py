from enum import Enum


class ComputationWorkload(Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'

def str_to_enum(value: str):
    match value:
        case ComputationWorkload.LOW.value:
            return ComputationWorkload.LOW
        case ComputationWorkload.MEDIUM.value:
            return ComputationWorkload.MEDIUM
        case ComputationWorkload.HIGH.value:
            return ComputationWorkload.HIGH
        case _:
            return ComputationWorkload.MEDIUM