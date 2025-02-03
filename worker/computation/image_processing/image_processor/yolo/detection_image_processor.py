import cv2 as cv
import numpy as np

from packages.enums import ComputeLoad
from worker.enums.loading_mode import LoadingMode
from worker.computation.image_processing.image_processor.yolo.yolo_image_processor import YOLOImageProcessor

class DetectionYOLOImageProcessor(YOLOImageProcessor):
    def __init__(self, compute_load: ComputeLoad, model_loading_mode: LoadingMode, model_paths: dict):
        super().__init__(compute_load, model_loading_mode, model_paths)

    def _draw_bounding_boxes_with_label(self, image, boxes, class_ids, confidences):
        """
        Draw  bounding boxes on an image.

        Parameters:
            image (np.ndarray): The image/frame on which to draw.
            bounding_boxes (np.ndarray): Array of bounding boxes in the format [x, y, w, h, r].
        """

        for xyxy, class_id, confidence in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = np.int0(xyxy)

            self._draw_bounding_boxes(image, x1, y1, x2, y2)
            self._draw_labels(image, class_id, confidence, x1, y1)

        return image

    def _draw_bounding_boxes(self, image, x1, y1, x2, y2):
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    def _draw_labels(self, image, class_id, confidence, x, y):
        label = f"{self._detector.class_names[int(class_id)]} {confidence:.2f}"

        # Adjust font scale and thickness for better readability
        text_size = cv.getTextSize(label, super().font, super().font_scale, super().font_thickness)[0]
        text_x, text_y = x, y - 10  # Position of the text

        # Draw a filled rectangle as background for the text
        cv.rectangle(image, (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0], text_y + 5), (255, 0, 0), cv.FILLED)

        # Draw the label text on top of the filled rectangle
        cv.putText(image, label, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX,
                   super().font_scale, (255, 255, 255), super().font_thickness)

    def _extract_bounding_boxes_info(self, inference_result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes =  inference_result[0].boxes.cpu().numpy().xyxy
        class_ids = inference_result[0].boxes.cls.cpu().numpy()
        confidences = inference_result[0].boxes.conf.cpu().numpy()
        return boxes, class_ids, confidences



