from enum import Enum


class WorkLoad(Enum):
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'

    @staticmethod
    def str_to_enum(value: str):
        match value:
            case WorkLoad.LOW.value:
                return WorkLoad.LOW
            case WorkLoad.MEDIUM.value:
                return WorkLoad.MEDIUM
            case WorkLoad.HIGH.value:
                return WorkLoad.HIGH
            case _:
                return WorkLoad.MEDIUM