from typing import Any

import cv2 as cv
from numpy import ndarray

class Video:
    def __init__(self, path: str):
        self.path = path
        self.video_capture = cv.VideoCapture(path)
        self.width = int(self.video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video_capture.get(cv.CAP_PROP_FPS)

    def read_frame(self) -> tuple[bool, ndarray | Any]:
        return self.video_capture.read()

    def release(self):
        self.video_capture.release()

    def isOpened(self) -> bool:
        return self.video_capture.isOpened()