from enum import Enum


class AgentType(Enum):
    NONE = 0
    TEST = 1
    ACTIVE_INFERENCE = 2
    REINFORCEMENT_LEARNING = 3
    HEURISTIC = 4