from dataclasses import dataclass, field

@dataclass
class WorkerStatistics:
    num_requested_tasks: int
    registration_time: float