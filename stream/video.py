import cv2 as cv


class Video:
    def __init__(self, path):
        self.path = path
        self.video_capture = cv.VideoCapture(path)
        self.width = int(self.video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video_capture.get(cv.CAP_PROP_FPS)

    def read_frame(self):
        return self.video_capture.read()

    def release(self):
        self.video_capture.release()

    def isOpened(self):
        return self.video_capture.isOpened()