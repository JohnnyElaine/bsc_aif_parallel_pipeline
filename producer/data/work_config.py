from packages.enums import ComputeLoad
from packages.enums import ComputeType

class WorkConfig:
    def __init__(self, compute_type: ComputeType, compute_load: ComputeLoad):
        self.compute_type = compute_type
        self.compute_load = compute_load
        self._work_distribution = {}






