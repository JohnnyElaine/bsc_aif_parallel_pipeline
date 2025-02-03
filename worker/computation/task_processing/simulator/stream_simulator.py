from abc import ABC

from worker.computation.task_processing.task_processor import TaskProcessor


class StreamSimulator(TaskProcessor, ABC):
    pass