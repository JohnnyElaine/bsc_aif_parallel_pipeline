from enum import Enum


class StreamType(Enum):
    SIMULATION = 0,
    LOCAL_MESSAGE = 1,
    UDP = 2
