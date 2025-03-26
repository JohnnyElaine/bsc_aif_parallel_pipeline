from enum import Enum


class AgentType(Enum):
    ACTIVE_INFERENCE = 0
    REINFORCEMENT_LEARNING = 1
    HEURISTIC = 2