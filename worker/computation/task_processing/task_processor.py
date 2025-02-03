from abc import ABC, abstractmethod

class TaskProcessor(ABC):
    @abstractmethod
    def stop(self):
        pass