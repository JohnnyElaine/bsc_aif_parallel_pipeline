import numpy as np
import cv2 as cv

from abc import abstractmethod, ABC

from worker.computation.image_processing.image_detector.yolo_detector import YoloDetector
from worker.computation.image_processing.image_processor.image_processor import ImageProcessor
from packages.enums import ComputeLoad
from worker.enums.loading_mode import LoadingMode


class YOLOImageProcessor(ImageProcessor, ABC):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2

    def __init__(self, compute_load: ComputeLoad, model_loading_mode: LoadingMode, model_paths: dict):
        self._model_loading_mode = model_loading_mode

        self._detector_low = YoloDetector(model_paths["low"])
        self._detector_medium = YoloDetector(model_paths["medium"])
        self._detector_high = YoloDetector(model_paths["high"])

        self._detector = self._set_detector(compute_load)

    def process_image(self, img):
        prediction_result = self._detector.predict_image(img)
        boxes, class_ids, confidences = self._extract_bounding_boxes_info(prediction_result)
        return self._draw_bounding_boxes_with_label(img, boxes, class_ids, confidences)
    
    def initialize(self):
        if self._model_loading_mode == LoadingMode.LAZY:
            self._detector.initialize()
        else:
            self._detector_low.initialize()
            self._detector_medium.initialize()
            self._detector_high.initialize()

    def change_detector(self, compute_load: ComputeLoad):
        self._set_detector(compute_load)

        if not self._detector.is_loaded():
            self._detector.initialize()

    def _set_detector(self, compute_load: ComputeLoad):
        match compute_load:
            case ComputeLoad.LOW:
                return self._detector_low
            case ComputeLoad.MEDIUM:
                return self._detector_medium
            case ComputeLoad.HIGH:
                return self._detector_high
            case _:
                return self._detector_low

    @abstractmethod
    def _draw_bounding_boxes_with_label(self, image, boxes, class_ids, confidences):
        pass


    @abstractmethod
    def _extract_bounding_boxes_info(self, inference_result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass