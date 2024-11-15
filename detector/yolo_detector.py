import torch
from ultralytics import YOLO


class YoloDector:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
