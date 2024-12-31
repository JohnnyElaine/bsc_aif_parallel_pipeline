import cv2 as cv
import numpy as np

from aif_edge_node.global_variables import GlobalVariables
from aif_edge_node.image_processing.image_detector.yolo_detector import YoloDetector
from aif_edge_node.image_processing.image_processor.image_processor import ImageProcessor


class DetectionImageProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()
        self.detector = YoloDetector(GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'detection' / 'yolo11n.pt')

    def _draw_bounding_boxes(self, image, bounding_boxes):
        """
        Draw  bounding boxes on an image.

        Parameters:
            image (np.ndarray): The image/frame on which to draw.
            bounding_boxes (np.ndarray): Array of bounding boxes in the format [x, y, w, h, r].
        """
        for xyxy in bounding_boxes:
            xyxy = np.int0(xyxy)
            x1, y1, x2, y2 = xyxy
            cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2) # Red color, thickness=2

        return image

    def _extract_bounding_boxes(self, inference_result) -> np.ndarray:
        return inference_result[0].boxes.cpu().numpy().xyxy



