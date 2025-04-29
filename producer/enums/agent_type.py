from enum import Enum


class AgentType(Enum):
    NONE = 0
    TEST = 1
    ACTIVE_INFERENCE = 2
    ACTIVE_INFERENCE_EXPERIMENTAL = 3
    REINFORCEMENT_LEARNING = 4
    HEURISTIC = 5