from enum import Enum


class AgentType(Enum):
    TEST = 0
    ACTIVE_INFERENCE = 1
    REINFORCEMENT_LEARNING = 2
    HEURISTIC = 3