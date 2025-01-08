from abc import ABC, abstractmethod

class StreamComputer(ABC):

    @abstractmethod
    def stop(self):
        pass