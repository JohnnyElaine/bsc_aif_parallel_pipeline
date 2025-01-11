import cv2 as cv
from numpy import ndarray

class Video:
    def __init__(self, path: str):
        self.path = path
        self.video_capture = cv.VideoCapture(path)
        self.width = int(self.video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video_capture.get(cv.CAP_PROP_FPS)
        self.frame_index = 0

    def read_frame(self) -> tuple[bool, ndarray, int]:
        index = self.frame_index
        ret, frame = self.video_capture.read()
        self.frame_index += 1

        return ret, frame, index

    def release(self):
        if self.is_opened():
            self.video_capture.release()

    def is_opened(self) -> bool:
        return self.video_capture.isOpened()