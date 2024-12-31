import logging.handlers
import numpy as np

from abc import ABC, abstractmethod


class ImageProcessor(ABC):
    def __init__(self):
        self.detector = None

    def process_image(self, img):
        prediction_result = self.detector.predict_image(img)
        bounding_boxes = self._extract_bounding_boxes(prediction_result)
        return self._draw_bounding_boxes(img, bounding_boxes)

    @abstractmethod
    def _draw_bounding_boxes(self, image, bounding_boxes):
        """
        Draw  bounding boxes on an image.

        Parameters:
            image (np.ndarray): The image/frame on which to draw.
            bounding_boxes (np.ndarray): Array of bounding boxes in the format specific to YOLO Mode
        """
        pass

    @abstractmethod
    def _extract_bounding_boxes(self, yolo_inference_result) -> np.ndarray:
        pass



