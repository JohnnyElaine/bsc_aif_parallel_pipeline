from enum import IntEnum

class SloStatus(IntEnum):
    OK = 0 # green
    WARNING = 1 # yellow
    CRITICAL = 2 # red - SLO is not satisfied