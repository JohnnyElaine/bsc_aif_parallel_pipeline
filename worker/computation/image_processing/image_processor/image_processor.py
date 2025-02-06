from abc import ABC, abstractmethod

class ImageProcessor(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def process_image(self, img):
        pass

    @abstractmethod
    def change_work_load(self, img):
        pass