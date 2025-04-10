import cv2 as cv
from numpy import ndarray

from producer.data.resolution import Resolution


class Video:
    def __init__(self, path: str):
        self.path = path
        self.video_capture = cv.VideoCapture(path)
        self.resolution = Resolution(int(self.video_capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(self.video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
        self.fps = round(self.video_capture.get(cv.CAP_PROP_FPS))
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

    def close(self):
        if not self.is_opened():
            return

        self.video_capture.release()

    @staticmethod
    def resize_frame(frame: ndarray, width: int, height: int) -> ndarray:
        return cv.resize(frame, (width, height))