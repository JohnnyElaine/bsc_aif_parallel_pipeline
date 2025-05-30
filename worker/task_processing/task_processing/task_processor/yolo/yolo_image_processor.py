import logging
from abc import abstractmethod, ABC

import cv2 as cv
import numpy as np

from packages.enums import InferenceQuality, LoadingMode
from worker.task_processing.task_processing.image_detector.yolo_detector import YoloDetector
from worker.task_processing.task_processing.task_processor.task_processor import TaskProcessor

log = logging.getLogger('task_processing')


class YOLOTaskProcessor(TaskProcessor, ABC):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2

    def __init__(self, work_load: InferenceQuality, model_loading_mode: LoadingMode, model_paths: dict):
        self._model_loading_mode = model_loading_mode

        self._detector_low = YoloDetector(model_paths['low'])
        self._detector_medium = YoloDetector(model_paths['medium'])
        self._detector_high = YoloDetector(model_paths['high'])

        self._detector = self._get_detector_by_work_load(work_load)

    @abstractmethod
    def _draw_bounding_boxes_with_label(self, image, boxes, class_ids, confidences):
        pass

    @abstractmethod
    def _extract_bounding_boxes_info(self, inference_result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def process(self, img):
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

    def change_work_load(self, work_load: InferenceQuality):
        self._detector = self._get_detector_by_work_load(work_load)

        if not self._detector.is_loaded():
            self._detector.initialize()

    def _get_detector_by_work_load(self, work_load: InferenceQuality):
        match work_load:
            case InferenceQuality.LOW:
                return self._detector_low
            case InferenceQuality.MEDIUM:
                return self._detector_medium
            case InferenceQuality.HIGH:
                return self._detector_high
            case _:
                return self._detector_low