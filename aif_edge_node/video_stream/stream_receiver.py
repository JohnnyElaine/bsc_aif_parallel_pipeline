from abc import ABC, abstractmethod

class StreamReceiver(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass