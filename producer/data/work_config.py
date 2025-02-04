from packages.enums import WorkLoad
from packages.enums import WorkType

class WorkConfig:
    def __init__(self, compute_type: WorkType, compute_load: WorkLoad):
        self.work_type = compute_type
        self.work_load = compute_load
        self._work_distribution = {}






