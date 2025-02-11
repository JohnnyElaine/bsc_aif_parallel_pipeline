import torch
import logging.handlers
from ultralytics import YOLO

log = logging.getLogger('task_handler')

class YoloDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.class_names = None
        self._device = None
        self._model = None

    def initialize(self):
        self._select_device()
        self._load_model()
        self._set_class_names()

    def predict_image(self, image):
        return self._model.predict(source=image, device=self._device, verbose=False)

    def is_loaded(self):
        return self._model is not None

    def _select_device(self):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info(f"Using {self._device}")

    def _load_model(self,):
        self._model = YOLO(self.model_path)
        log.info(f'Successfully Loaded model {self._model.info()}, from {self.model_path}') # TODO does not get printed when switching work load

        self.class_names = self._model.names

    def _set_class_names(self):
        self.class_names = self._model.names

