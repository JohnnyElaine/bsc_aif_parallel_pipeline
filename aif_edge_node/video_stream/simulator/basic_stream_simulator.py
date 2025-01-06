import logging
import time
import cv2 as cv
import numpy as np

from aif_edge_node.video_stream.stream_receiver import StreamReceiver
from aif_edge_node.video_stream.video import Video
from aif_edge_node.image_processing.image_processor.image_processor import ImageProcessor

log = logging.getLogger("aif_edge_node")


class BasicStreamSimulator(StreamReceiver):
    def __init__(self, image_processor: ImageProcessor, vid_path, show_result=True):
        self.video = Video(vid_path)
        self._target_frame_time = 1 / self.video.fps
        self.max_display_width = 1280  # TODO: make dynamic to suit input video
        self.max_display_height = 720
        self.display_width, self.display_height = self._scale_display_dimensions()
        self.image_processor = image_processor
        self.is_running = False

    def start(self):
        self.is_running = True

        if not self.video.is_opened():
            raise IOError(f'Unable to open input video file. Path: {self.video.path}')

        self._play_video()

        self.stop()

    def stop(self):
        """
        Stops the video video_stream, releases the video capture and destroys all openCV windows
        :return:
        """
        if not self.is_running:
            return

        self.is_running = False
        self.video.release()
        cv.destroyAllWindows()

    def _scale_display_dimensions(self):
        """
        Scales the dimensions of the video stream to fit within the set max_height and max_width.

        Returns
        -------
        display_width
            The new width of the video stream.
        display_height
            The new height of the video stream.
        """
        display_width = self.video.width
        display_height = self.video.height

        if display_width > self.max_display_width or display_height > self.max_display_height:
            scale_ratio = min(self.max_display_width / display_width, self.max_display_height / display_height)
            display_width = int(display_width * scale_ratio)
            display_height = int(display_height * scale_ratio)

        return display_width, display_height

    def _play_video(self):
        prev_frame_time = time.perf_counter()

        while self.is_running:
            iteration_start_time = time.perf_counter()

            ret, frame = self.video.read_frame()
            if not ret:
                log.debug("End of video stream or error reading frame.")
                break

            frame = self._process_image(frame)

            current_time = time.perf_counter()
            fps = int(1 / (current_time - prev_frame_time))
            prev_frame_time = current_time

            self._display_image(frame, fps)

            # Exit if the 'q' key is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            self._enforce_target_fps(iteration_start_time)


    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Takes a frame from an original source and processes it using:
        YOLOv11, OpenCV Image resize. Returning the processed frame.

        Parameters
        ----------
        image

        Returns
        -------
        The processed and (optionally) resized frame.
        """
        img = self.image_processor.process_image(image)
        img = cv.resize(img, (self.display_width, self.display_height))

        return img

    def _display_image(self, image, fps):
        self._display_fps(image, fps)
        cv.imshow('Video', image)

    def _enforce_target_fps(self, iteration_start_time: float):
        iteration_duration = time.perf_counter() - iteration_start_time
        wait_time = max(self._target_frame_time - iteration_duration, 0)
        if wait_time > 0:
            time.sleep(wait_time)

    @staticmethod
    def _display_fps(frame, fps):
        # Overlay FPS text on the frame
        fps_text = f"FPS: {fps}"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 0, 255)
        thickness = 2
        position = (10, 30)  # Top-left corner

        cv.putText(frame, fps_text, position, font, font_scale, color, thickness)




