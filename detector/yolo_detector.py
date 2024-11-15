import logging
import torch
from ultralytics import YOLO


class YoloDetector:
    def __init__(self):
        self.device = self._select_device()
        self.model = self._load_model()
        self._load_model()

    def handle_frame(self, frame):
        self.model.track(source=frame, device=self.device, tracker='./trackers/bytetrack.yaml', show=True)

    @staticmethod
    def _select_device():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')
        return device

    @staticmethod
    def _load_model():
        model = YOLO("/checkpoints/models/detection/yolo11n.pt")
        print(f'Loaded Model: {model}')
        return model


