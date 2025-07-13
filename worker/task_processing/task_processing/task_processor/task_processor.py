from abc import ABC, abstractmethod

class TaskProcessor(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def process(self, img):
        pass

    @abstractmethod
    def change_inference_quality(self, img):
        pass