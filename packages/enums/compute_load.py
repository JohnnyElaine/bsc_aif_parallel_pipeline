from enum import Enum


class ComputeLoad(Enum):
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'

def str_to_enum(value: str):
    match value:
        case ComputeLoad.LOW.value:
            return ComputeLoad.LOW
        case ComputeLoad.MEDIUM.value:
            return ComputeLoad.MEDIUM
        case ComputeLoad.HIGH.value:
            return ComputeLoad.HIGH
        case _:
            return ComputeLoad.MEDIUM