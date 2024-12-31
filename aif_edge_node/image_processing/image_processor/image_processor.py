from abc import ABC, abstractmethod

class ImageProcessor(ABC):

    @abstractmethod
    def process_image(self, img):
        pass