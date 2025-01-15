from enum import Enum


class ComputeType(Enum):
    NONE = 'NONE'
    YOLO_OBB = 'YOLO_OBB'
    YOLO_DETECTION = 'YOLO_DETECTION'

def str_to_enum(value: str):
    match value:
        case ComputeType.YOLO_OBB.value:
            return ComputeType.YOLO_OBB
        case ComputeType.YOLO_DETECTION.value:
            return ComputeType.YOLO_DETECTION
        case _:
            return ComputeType.NONE