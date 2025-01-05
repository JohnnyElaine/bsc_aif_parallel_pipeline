import numpy as np
import cv2 as cv

from abc import abstractmethod
from aif_edge_node.image_processing.image_processor.image_processor import ImageProcessor

class YOLOImageProcessor(ImageProcessor):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2

    def __init__(self):
        self.detector = None

    def process_image(self, img):
        prediction_result = self.detector.predict_image(img)
        boxes, class_ids, confidences = self._extract_bounding_boxes_info(prediction_result)
        return self._draw_bounding_boxes_with_label(img, boxes, class_ids, confidences)

    @abstractmethod
    def _draw_bounding_boxes_with_label(self, image, boxes, class_ids, confidences):
        pass


    @abstractmethod
    def _extract_bounding_boxes_info(self, inference_result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass