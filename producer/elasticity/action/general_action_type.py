from enum import IntEnum

class GeneralActionType(IntEnum):
    """Must be numerated from 0-n"""
    NONE = 0
    INCREASE_RESOLUTION = 1
    DECREASE_RESOLUTION = 2
    INCREASE_FPS = 3
    DECREASE_FPS = 4
    INCREASE_INFERENCE_QUALITY = 5
    DECREASE_INFERENCE_QUALITY = 6
