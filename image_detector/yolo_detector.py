import torch
import logging.handlers
from ultralytics import YOLO

log = logging.getLogger("my_app")


class YoloDetector:
    def __init__(self, model_path: str):
        self.device = self._select_device()
        self.model = self._load_model(model_path)

    def predict_frame(self, frame):
        return self.model(source=frame, device=self.device, verbose=False)
        #self.model.track(source=frame, device=self.device, tracker='./trackers/bytetrack.yaml', show=True)

    @staticmethod
    def _select_device():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.debug(f"Using {device}")
        return device

    @staticmethod
    def _load_model(path: str):
        model = YOLO(path)
        log.debug(f'Loaded model {model.info()}')
        return model


