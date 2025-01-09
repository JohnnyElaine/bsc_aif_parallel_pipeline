from enum import Enum


class ComputationType(Enum):
    OBB = 'obb'
    DETECTION = 'detection'
    NONE = 'none'

def str_to_computation_type(str: str):
    match str:
        case "obb":
            return ComputationType.OBB
        case "detection":
            return ComputationType.DETECTION
        case _:
            return ComputationType.NONE