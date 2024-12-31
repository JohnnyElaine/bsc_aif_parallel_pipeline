import torch
import logging.handlers
from ultralytics import YOLO

log = logging.getLogger("aif_edge_node")

class YoloDetector:
    def __init__(self, model_path: str):
        self.device = self._select_device()
        self.model = self._load_model(model_path)
        self.class_names = self.model.names

    def predict_image(self, image):
        return self.model.predict(source=image, device=self.device, verbose=False)

    @staticmethod
    def _select_device():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.debug(f"Using {device}")
        return device

    @staticmethod
    def _load_model(path: str):
        model = YOLO(path)
        log.debug(f'Successfully Loaded model {model.info()}, from {path}')
        return model


