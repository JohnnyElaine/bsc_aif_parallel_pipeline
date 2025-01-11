from enum import Enum


class ComputationType(Enum):
    NONE = 'none'
    YOLO_OBB = 'yolo_obb'
    YOLO_DETECTION = 'yolo_detection'

def str_to_enum(value: str):
    match value:
        case ComputationType.YOLO_OBB.value:
            return ComputationType.YOLO_OBB
        case ComputationType.YOLO_DETECTION.value:
            return ComputationType.YOLO_DETECTION
        case _:
            return ComputationType.NONE