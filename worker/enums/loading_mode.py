from enum import Enum


class LoadingMode(Enum):
    LAZY = 0
    EAGER = 1

    @staticmethod
    def int_to_enum(value: int):
        match value:
            case LoadingMode.LAZY.value:
                return LoadingMode.LAZY
            case LoadingMode.EAGER.value:
                return LoadingMode.EAGER
            case _:
                return LoadingMode.LAZY

