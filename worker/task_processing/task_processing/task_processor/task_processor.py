from abc import ABC, abstractmethod

class TaskProcessor(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def process(self, img):
        pass

    @abstractmethod
    def change_work_load(self, img):
        pass