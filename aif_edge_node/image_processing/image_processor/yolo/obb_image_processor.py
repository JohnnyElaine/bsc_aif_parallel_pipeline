import cv2 as cv
import numpy as np

from aif_edge_node.global_variables import GlobalVariables
from aif_edge_node.image_processing.image_detector.yolo_detector import YoloDetector
from aif_edge_node.image_processing.image_processor.yolo.yolo_image_processor import YOLOImageProcessor

class OBBYOLOImageProcessor(YOLOImageProcessor):
    def __init__(self):
        super().__init__()

        self.detector = YoloDetector(GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'obb' / 'yolo11n-obb.pt')

    def initialize(self):
        self._detector.initialize()

    def _draw_bounding_boxes_with_label(self, image, boxes, class_ids, confidences):
        """
        Draw (oriented bounding boxes on an image.

        Parameters:
            image (np.ndarray): The image/frame on which to draw.
            bounding_boxes (np.ndarray): Array of bounding boxes in the format [x, y, w, h, r].
        """
        for xywhr, class_id, confidence in zip(boxes, class_ids, confidences):
            x, y, w, h, r = xywhr

            # Prepare for cv2.boxPoints
            rect = ((x, y), (w, h), np.degrees(r))  # Convert angle to degrees
            box = cv.boxPoints(rect)  # Get the 4 vertices of the rotated rectangle
            box = np.int0(box)  # Convert vertex coordinates to integers

            # Draw the bounding box
            self._draw_bounding_boxes(image, box)

            # Calculate the bottom-right corner of the OBB (max x, max y)
            bottom_right = tuple(np.max(box, axis=0))

            # Draw the label at the bottom-right corner
            self._draw_labels(image, class_id, confidence, bottom_right[0], bottom_right[1])

        return image

    def _draw_bounding_boxes(self, image, contours):
        cv.drawContours(image, [contours], 0, (0, 0, 255), 2)  # Red color, thickness=2

    def _draw_labels(self, image, class_id, confidence, x, y):
        label = f"{self.detector.class_names[int(class_id)]} {confidence:.2f}"

        # Adjust font scale and thickness for better readability
        text_size = cv.getTextSize(label, super().font, super().font_scale, super().font_thickness)[0]
        text_x, text_y = x - text_size[0], y + 5  # Position the text at the bottom-right corner

        # Draw a filled rectangle as background for the text
        cv.rectangle(image, (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0], text_y + 5), (0, 0, 255), cv.FILLED)

        # Draw the label text on top of the filled rectangle
        cv.putText(image, label, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX,
                   super().font_scale, (255, 255, 255), super().font_thickness)

    def _extract_bounding_boxes_info(self, inference_result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = inference_result[0].obb.cpu().numpy().xywhr
        class_ids = inference_result[0].obb.cls.cpu().numpy()
        confidences = inference_result[0].obb.conf.cpu().numpy()
        return boxes, class_ids, confidences



