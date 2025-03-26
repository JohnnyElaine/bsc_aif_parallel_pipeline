from enum import Enum

class ActionType(Enum):
    """Must be numerated from 0-n"""
    INCREASE_RESOLUTION = 0
    DECREASE_RESOLUTION = 1
    INCREASE_FPS = 2
    DECREASE_FPS = 3
    INCREASE_WORK_LOAD = 4
    DECREASE_WORK_LOAD = 5
    DO_NOTHING = 6