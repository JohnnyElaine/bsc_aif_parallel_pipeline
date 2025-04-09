from enum import Enum

class SloStatus(Enum):
    OK = 0 # green
    WARNING = 1 # yellow
    CRITICAL = 2 # red - SLO is not satisfied