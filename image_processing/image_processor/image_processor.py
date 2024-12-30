import logging.handlers
import cv2 as cv
import numpy as np

from image_processing.image_detector.yolo_detector import YoloDetector

log = logging.getLogger("aif_edge_node_log")


class ImageProcessor:
    def __init__(self):
        self.detector = YoloDetector("./checkpoints/models/obb/yolo11n-obb.pt")

    def process_image(self, img):
        prediction_result = self.detector.predict_image(img)
        bounding_boxes = self._extract_obb(prediction_result)
        return self._draw_obb(img, bounding_boxes)

    @staticmethod
    def _draw_obb(image, bounding_boxes):
        """
        Draw oriented bounding boxes on an image.

        Parameters:
            image (np.ndarray): The image/frame on which to draw.
            bounding_boxes (np.ndarray): Array of bounding boxes in the format [x, y, w, h, r].
        """
        for xywhr in bounding_boxes:
            x, y, w, h, r = xywhr  # Do NOT convert to int prematurely

            # Prepare for cv2.boxPoints
            rect = ((x, y), (w, h), np.degrees(r))  # Convert angle to degrees
            box = cv.boxPoints(rect)  # Get the 4 vertices of the rotated rectangle
            box = np.int0(box)  # Convert vertex coordinates to integers

            # Draw the oriented bounding box
            cv.drawContours(image, [box], 0, (0, 0, 255), 2)  # Red color, thickness=2

        return image

    @staticmethod
    def _extract_obb(yolo_inference_result) -> np.ndarray:
        return yolo_inference_result[0].obb.cpu().numpy().xywhr



