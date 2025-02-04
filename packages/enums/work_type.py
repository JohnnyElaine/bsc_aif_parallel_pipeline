from enum import Enum


class WorkType(Enum):
    NONE = 'NONE'
    YOLO_OBB = 'YOLO_OBB'
    YOLO_DETECTION = 'YOLO_DETECTION'

    @staticmethod
    def str_to_enum(value: str):
        match value:
            case WorkType.YOLO_OBB.value:
                return WorkType.YOLO_OBB
            case WorkType.YOLO_DETECTION.value:
                return WorkType.YOLO_DETECTION
            case _:
                return WorkType.NONE
