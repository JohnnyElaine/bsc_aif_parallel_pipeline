from dataclasses import dataclass

@dataclass
class WorkerStatistics:
    num_requested_tasks: int
    registration_time: float