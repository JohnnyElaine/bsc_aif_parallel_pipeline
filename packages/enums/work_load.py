from enum import Enum


class WorkLoad(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    @staticmethod
    def int_to_enum(value: int):
        match value:
            case WorkLoad.LOW.value:
                return WorkLoad.LOW
            case WorkLoad.MEDIUM.value:
                return WorkLoad.MEDIUM
            case WorkLoad.HIGH.value:
                return WorkLoad.HIGH
            case _:
                return WorkLoad.MEDIUM


w = WorkLoad.LOW