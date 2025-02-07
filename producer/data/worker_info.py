from dataclasses import dataclass


@dataclass
class WorkerInfo:
        preferred_num_of_tasks = 1
        instruction_backlog = list() # queue